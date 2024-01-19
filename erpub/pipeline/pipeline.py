from functools import partial
import glob
import logging
import os
import inspect
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from erpub.pipeline.blocking import naive_all_pairs
from erpub.pipeline.matching import jaccard_similarity

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

DEFAULT_ATTRIBUTES = [
    "paper_title",
    "author_names",
    "publication_venue",
    "year_of_publication",
]


class Pipeline:
    def __init__(
        self,
        file_dir: str,
        preprocess_data_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        blocking_fn: Callable[[Sequence], np.ndarray] = naive_all_pairs,
        matching_fns: dict[
            str,
            Callable[[str, str], float]
            | Callable[[str, str, dict[str, np.ndarray]], float],
        ] = {attr: jaccard_similarity for attr in DEFAULT_ATTRIBUTES},
        embeddings_for_matching: str | None = None,
    ):
        self.df = Pipeline._load_data(file_dir)
        if embeddings_for_matching is None and any(
            Pipeline._requires_embedding_table(f) for f in matching_fns.values()
        ):
            raise ValueError(
                "Missing required parameter 'embeddings_for_matching' as one of the matching functions require them"
            )
        self.embedding_table = (
            Pipeline._get_embedding_table(embeddings_for_matching)
            if embeddings_for_matching
            else None
        )
        self.preprocess_data_fn = preprocess_data_fn
        self.blocking_fn = blocking_fn
        self.matching_fns = matching_fns
        self.matching_fns_vec: dict[
            str, Callable[[Sequence[str], Sequence[str]], float]
        ] = {attr: np.vectorize(partial(match_f, embedding_table=self.embedding_table)) if self._requires_embedding_table(match_f) else np.vectorize(match_f) for attr, match_f in matching_fns.items()}
        self.matched_pairs: np.ndarray | None = None
        logging.info("Pipeline initialized")

    @staticmethod
    def _load_data(file_dir: str) -> pd.DataFrame:
        """Load all csv files in the specified directory into a single pandas DataFrame.

        Parameters
        ----------
        file_dir : str
            Directory path containing the csv files created by data_preparation.py

        Returns
        ----------
        df : pd.DataFrame
            All the csv files contained in file_dir as a single pandas DataFrame
            with a column "dataset" containing the name of the file.
        """
        all_files = glob.glob(os.path.join(file_dir, "*.csv"))
        logging.info(f"The pipeline will be built with these files: {all_files}")
        df = pd.concat(
            (pd.read_csv(f).assign(dataset=f) for f in all_files), ignore_index=True
        ).astype(str)
        logging.info("Loaded csv successfully into pandas dataframe")
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

    def _write_matched_entities_csv(
        self, matched_pairs: np.ndarray, dir_name: str, similarity_threshold: float
    ) -> None:
        """Writes for each match between entities from different datasets
        an entry to the match_dir.

        Parameters
        ----------
        matched_pairs : ndarray(dtype=int64, ndim=2)
            Array of size matched_pairs x 2 containing the indices for each pair as a row.
        dir_name : str
            Directory path where the matched_entities.csv and pipeline_settings.txt will be placed.
        """
        unique_datasets = self.df.dataset.unique()
        match_ids: list[dict[str, str]] = [
            self.df.iloc[match].set_index("dataset").paper_id.to_dict()
            for match in matched_pairs
            if len(self.df.iloc[match].dataset.unique()) == len(unique_datasets)
        ]
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(match_ids).to_csv(
            os.path.join(dir_name, "matched_entities.csv"), index=False
        )
        with open(
            os.path.join(dir_name, "pipeline_settings.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(f"Blocking function: {self.blocking_fn.__name__}\n")
            for attr, f in self.matching_fns.items():
                file.write(f"Matching function for attribute {attr}: {f.__name__}\n")
            file.write(f"Similarity threshold: {similarity_threshold}\n")

    @staticmethod
    def _cluster_matched_entities(
        matched_pairs: Iterable[tuple[int, int]]
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
        df = self.df.copy()
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

    def _get_similarity_scores(self, pairs_to_match: np.ndarray) -> np.ndarray:
        """Calculates the average similarity score over matching_fns.
        Parameters
        ----------
        pairs_to_match : ndarray(dtype=int64, ndim=2)
            Array of size #pairs x 2 containing the indices for each pair as a row.
        Returns
        ----------
        similarity_score : ndarray(dtype=float64, ndim=0)
            Array containing the similarity scores for each pair.
        """
        similarity_scores = []
        for attr, match_f in self.matching_fns_vec.items():
            attr_df = self.df[attr]
            a = attr_df.iloc[pairs_to_match[:, 0]]
            b = attr_df.iloc[pairs_to_match[:, 1]]
            similarity_scores.append(match_f(a, b))
        return np.mean(similarity_scores, axis=0)

    def run(self, dir_name: str, similarity_threshold: float) -> None:
        """Execute the entity resolution pipeline.

        Parameters
        ----------
        dir_name : str
            Directory path where the matched_entities.csv and pipeline_settings.txt will be placed.
        """
        if self.preprocess_data_fn:
            logging.info("Data will be preprocessed")
            self.df = self.preprocess_data_fn(self.df)
        else:
            logging.info("Preprocessing has been skipped")

        logging.info(
            f"Create list of pairs through blocking function {self.blocking_fn.__name__}"
        )
        pairs_to_match = self.blocking_fn(self.df)
        logging.info(f"Calculate similarities of {len(pairs_to_match)} pairs")
        for attr, f in self.matching_fns.items():
            logging.info(f"Attribute '{attr}' is matched using function {f.__name__}")
        similarity_scores = self._get_similarity_scores(pairs_to_match)
        self.matched_pairs = pairs_to_match[
            similarity_scores > similarity_threshold
        ]
        logging.info("Writing the matched paper_ids to directory {dir_name}")
        self._write_matched_entities_csv(self.matched_pairs, dir_name, similarity_threshold)

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
            clusters = Pipeline._cluster_matched_entities(self.matched_pairs)
            logging.info("Writing the resolved dataset to directory {dir_name}")
            self._write_resolved_data(clusters, dir_name)
