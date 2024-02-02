import glob
import inspect
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import dask.dataframe as dd
from dask.distributed import Client

from erpub.pipeline.blocking_dask import naive_all_pairs_dask
from erpub.pipeline.matching_dask import jaccard_similarity

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

DEFAULT_ATTRIBUTES = [
    "paper_title",
    "author_names",
    "publication_venue",
    "year_of_publication",
]


class DaskPipeline:
    def __init__(
        self,
        file_dir: str,
        matching_fns: dict,
        similarity_threshold: float,
        preprocess_data_fn: Callable[[dd.DataFrame], dd.DataFrame] | None = None,
        blocking_fn: Callable[[dd.DataFrame], None] = naive_all_pairs_dask,
        embeddings_for_matching: str | None = None,
        verbose: bool = True,
        client: Client | None = None,
    ):
        self.client = client if client is not None else Client()
        if not verbose:
            logging.disable(logging.INFO)

        self.original_df = DaskPipeline._load_data(file_dir)
        self.preprocess_data_fn = preprocess_data_fn
        if preprocess_data_fn:
            logging.info("Data will be preprocessed")
            self.df = preprocess_data_fn(self.original_df.copy())
        else:
            logging.info("Preprocessing has been skipped")
            self.df = self.original_df.copy()
        """ if embeddings_for_matching is None and any(
            DaskPipeline._requires_embedding_table(f) for f in matching_fns.values()
        ):
            raise ValueError(
                "Missing required parameter 'embeddings_for_matching' as one of the matching functions require them"
            ) """

        self.matching_fns = matching_fns
        self.similarity_threshold = similarity_threshold
        """ self.embedding_table = (
            self._get_embedding_table(embeddings_for_matching)
            if embeddings_for_matching
            else None
        ) """

        self.blocking_fn = blocking_fn
        self.matching_fns = matching_fns
        logging.info("Pipeline initialized")

    @staticmethod
    def _load_data(file_dir: str) -> dd.DataFrame:
        """Load all csv files in the specified directory into a single Dask DataFrame.

        Parameters
        ----------
        file_dir : str
            Directory path containing the csv files created by data_preparation.py

        Returns
        ----------
        df : dd.DataFrame
            All the csv files contained in file_dir as a single Dask DataFrame
            with a column "dataset" containing the name of the file.
        """
        all_files = glob.glob(os.path.join(file_dir, "*.csv"))
        logging.info(f"The pipeline will be built with these files: {all_files}")

        dfs = []
        for file in all_files:
            temp_df = dd.read_csv(file).astype(str)
            temp_df["dataset"] = os.path.basename(file)
            dfs.append(temp_df)

        df = dd.concat(dfs, axis=0, ignore_index=True)

        logging.info("Loaded csv successfully into Dask dataframe")
        return df

    @staticmethod
    def _get_embedding_table(path: str) -> defaultdict[str, np.ndarray]:
        """Build the embedding table as default dict with a random ndarray as default

        Parameters
        ----------
        path: str
            The path of the embeddings txt file (e.g. "glove.6B.50d.txt")

        Returns
        ----------
        embeddings_dict: defaultdict[str, np.ndarray]
            The embeddings mapping from word to vector
            and on unkown word returning a random vector
        """
        embeddings_dict = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        default_vector = np.random.rand(*vector.shape)
        return defaultdict(lambda: default_vector, embeddings_dict)

    @staticmethod
    def _requires_embedding_table(match_f: Callable) -> bool:
        "Returns True if the passed match_f requires embedding_table as it's argument, else False"
        return "embedding_table" in inspect.signature(match_f).parameters

    # lambda replacement for dask
    def _extract_match_info(self, index_a, index_b):
        return {"index_a": index_a, "index_b": index_b}

    def _apply_extract_match_info(self, df):
        """Applies _extract_match_info method across DataFrame rows."""
        return df.apply(
            lambda row: self._extract_match_info(row["index_a"], row["index_b"]),
            axis=1,
            meta=object,
        )

    def _write_matched_entities_csv(
        self, matched_pairs: dd.DataFrame, dir_name: str
    ) -> None:
        """Writes the matched entities to a CSV file.

        Parameters
        ----------
        matched_pairs : dd.DataFrame
            Dask DataFrame containing the matched pairs.
        dir_name : str
            Directory path where the matched_entities.csv will be put.
        """
        # Ensure the directory exists
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        # Apply the extract match info operation using map_partitions
        matched_info = matched_pairs.map_partitions(
            self._apply_extract_match_info,
            meta=object,
        )
        # Compute the Dask DataFrame and write the output to a CSV file
        matched_info.compute().to_csv(
            os.path.join(dir_name, "matched_entities.csv"), index=False
        )

    def calculate_attribute_similarities(self) -> dd.DataFrame:
        """
        Calculate similarity scores for each attribute and combine them.

        Parameters:
        - match_fn: Function to compute similarity between two series.

        Returns:
        - A Dask DataFrame with similarity scores for each attribute.
        """
        similarity_scores = []

        for attribute in DEFAULT_ATTRIBUTES:
            score = self.df[attribute].apply(
                lambda x: jaccard_similarity(x, attribute), meta=float
            )
            """ score = self.df.map_partitions(
                lambda partition: match_fn(partition[attribute], partition[attribute]),
                meta=float,
            ) """
            similarity_scores.append(score)

        # Combine scores into a DataFrame
        combined_scores = dd.concat(similarity_scores, axis=1)
        combined_scores.columns = DEFAULT_ATTRIBUTES

        return combined_scores

    def aggregate_similarity_scores(self, similarity_scores: dd.DataFrame) -> dd.Series:
        """
        Aggregate similarity scores across attributes.

        Parameters:
        - similarity_scores: Dask DataFrame of similarity scores for each attribute.

        Returns:
        - A Dask Series with aggregated similarity scores.
        """
        aggregated_scores = similarity_scores.mean(axis=1)
        return aggregated_scores

    @staticmethod
    def _cluster_matched_entities(
        matched_pairs: Iterable[tuple[int, int]],
    ) -> list[set[int]]:
        """Clusters the matched_pairs.

        Parameters
        ----------
        matched_pairs : ndarray(dtype=int64, ndim=2)
            Array of size matched_pairs x 2 containing the indices for each pair as a row.

        Returns
        ----------
        clusters : list[set[int]]
            A list of clusters where each cluster is a set of the indices.
        """
        clusters: list[set[int]] = []
        for a, b in matched_pairs:
            updated_cluster = False
            for cluster in clusters:
                if a in cluster or b in cluster:
                    cluster.update((a, b))
                    updated_cluster = True
                    break

            if not updated_cluster:
                clusters.append({a, b})

        return clusters

    @staticmethod
    def _apply_matching_fn(partition, match_fn, embedding_table=None):
        """Applies matching function on a partition."""
        return partition.apply(lambda x: match_fn(x, embedding_table))

    def get_matched_pairs(self, similarity_scores: dd.DataFrame) -> dd.DataFrame:
        """Identify matched pairs above the similarity threshold."""
        matched_indices = similarity_scores[
            similarity_scores >= self.similarity_threshold
        ]
        return matched_indices

    def _write_resolved_data(self, clusters: list[set[int]], dir_name: str) -> None:
        """Based on the clusters removes all found duplicates from all datasets
        and writes the updated csv to the resolved_files_dir.

        Parameters
        ----------
        clusters : list(set[int])
            A list of clusters where each cluster is a set of the indices.
        dir_name : str
            Directory path where the resolved csv files will be put.
        """
        df = self.original_df
        all_columns = ["paper_id"] + DEFAULT_ATTRIBUTES
        for cluster in clusters:
            cluster_list = list(cluster)
            df.loc[cluster_list, all_columns] = df.loc[
                cluster_list[0], all_columns
            ].values
        df = df.drop_duplicates()
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        for dataset in df.dataset.unique():
            dataset_df = df[df["dataset"] == dataset][all_columns]
            dataset_df.to_csv(
                os.path.join(dir_name, "resolved_" + dataset.split("/")[-1]),
                index=False,
            )

    def run(self, dir_name: str, similarity_threshold: float) -> float:
        """Execute the entity resolution pipeline.

        Parameters
        ----------
        dir_name : str
            Directory path where the matched_entities.csv and pipeline_settings.txt will be placed.
        similarity_threshold : float
            Similarity threshold used for matching.

        Returns
        ----------
        pipeline_execution_time : float
            The pipeline execution time in seconds.
        """
        logging.info(
            f"Create blocks through blocking function {self.blocking_fn.__name__}"
        )
        pipeline_start_time = time.time()
        self.blocking_fn(self.df)
        logging.info(f"Amount of different blocks: {self.df['block'].nunique()}")

        sim = self.calculate_attribute_similarities()

        aggregated_scores = self.aggregate_similarity_scores(sim)

        self.matched_pairs = self.get_matched_pairs(
            aggregated_scores,
        )
        pipeline_execution_time = time.time() - pipeline_start_time
        logging.info(f"Pipeline executed in {pipeline_execution_time} seconds.")
        logging.info(f"Writing the matched paper_ids to directory {dir_name}")
        self._write_matched_entities_csv(self.matched_pairs, dir_name)
        return pipeline_execution_time

    def resolve(self, dir_name: str) -> None:
        """Resolves the matched entities and writing the new data to dir_name.

        Parameters
        ----------
        dir_name : str
            Directory path where the resolved csv files will be put.
        """
        if self.matched_pairs is None:
            logging.warning(
                "Before resolving the entities you need to first have a succesful pipeline run"
            )
        else:
            logging.info("Clustering matched entities")
            clusters = DaskPipeline._cluster_matched_entities(self.matched_pairs)
            logging.info(f"Writing the resolved dataset to directory {dir_name}")
            self._write_resolved_data(clusters, dir_name)
