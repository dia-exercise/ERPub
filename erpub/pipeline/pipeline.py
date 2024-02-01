import glob
import inspect
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

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
        blocking_fn: Callable[[pd.DataFrame], None] = naive_all_pairs,
        matching_fns: dict[
            str,
            Callable[[str, str], float]
            | Callable[[pd.Series, defaultdict[str, np.ndarray]], np.ndarray],
        ] = {attr: jaccard_similarity for attr in DEFAULT_ATTRIBUTES},
        embeddings_for_matching: str | None = None,
        verbose: bool = True,
    ):
        """Initialize the Entity Resolution Pipeline.

        Parameters
        ----------
        file_dir : str
            Directory path containing the csv files created by data_preparation.py.
        preprocess_data_fn : Callable[[pd.DataFrame], pd.DataFrame] | None, optional
            Function to preprocess the input data, by default None.
        blocking_fn : Callable[[pd.DataFrame], None], optional
            Function to perform blocking and create blocks, by default naive_all_pairs.
        matching_fns : dict, optional
            Dictionary of attribute names and matching functions, by default using Jaccard similarity
            for default attributes ("paper_title", "author_names", "publication_venue", "year_of_publication").
        embeddings_for_matching : str | None, optional
            Path to the embeddings txt file (e.g., "glove.6B.50d.txt") used for matching functions
            requiring embeddings, by default None.
        verbose : bool, optional
            Whether to output logging information, by default True

        Raises
        ----------
        ValueError
            If `embeddings_for_matching` is missing, but required by matching functions.
        """
        if not verbose:
            logging.disable(logging.INFO)
        self.original_df = Pipeline._load_data(file_dir)
        self.preprocess_data_fn = preprocess_data_fn
        if preprocess_data_fn:
            logging.info("Data will be preprocessed")
            self.df = preprocess_data_fn(self.original_df.copy())
        else:
            logging.info("Preprocessing has been skipped")
            self.df = self.original_df.copy()
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
        self.blocking_fn = blocking_fn
        self.matching_fns = matching_fns
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

    @staticmethod
    def _requires_pandas_series(fn: Callable) -> bool:
        return (
            next(iter(inspect.signature(fn).parameters.values())).annotation
            == pd.Series
        )

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
        similarity_threshold : float
            Similarity threshold used for matching.
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
            if self.preprocess_data_fn:
                file.write(f"Preprocessing function: {self.preprocess_data_fn.__name__}\n")
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

    def _get_similarity_matrices(self) -> dict[str, np.ndarray]:
        """Calculates a dict with the average similarity matrix over matching_fns for each block.
        Returns
        ----------
        similarity_matrices : dict[str, ndarray(dtype=float64, ndim=2)]
            Dict with similarity matrices for each cluster of size NxN where N is the elements in the block.
        """
        similarity_matrices = {}
        for block_id, block_df in self.df.groupby("block"):
            matrix_lst: list[np.ndarray] = [
                (
                    match_f(block_df[attr], self.embedding_table)  # type: ignore
                    if Pipeline._requires_embedding_table(match_f)
                    and Pipeline._requires_pandas_series(match_f)
                    else self._vectorize_simple_function(block_df[attr], match_f)  # type: ignore
                )
                for attr, match_f in self.matching_fns.items()
            ]
            sim_matrix_block = np.mean(
                matrix_lst, axis=0
            )  # Average of the matching functions
            similarity_matrices[block_id] = sim_matrix_block
        return similarity_matrices

    def _vectorize_simple_function(
        self, data: pd.Series, fn: Callable[[str, str], float]
    ) -> np.ndarray:
        """Vectorizes a simple function over the elements of a pandas Series.

        Parameters
        ----------
        data : pd.Series
            The pandas Series containing the data.
        fn : Callable[[str, str], float]
            The function to be vectorized.

        Returns
        ----------
        similarity_matrix : ndarray(dtype=float64, ndim=2)
            Resulting matrix after applying the function to all pairs of elements in the Series.
            It's of size NxN where N is the length of the Series.
        """
        n = len(data)
        similarity_matrix = np.zeros((n, n))
        indices_a, indices_b = np.triu_indices(n)
        a = data.iloc[indices_a]
        b = data.iloc[indices_b]
        similarity_matrix[indices_a, indices_b] = np.vectorize(fn)(a, b)
        return similarity_matrix

    def _get_matched_pairs(
        self, similarity_scores: dict[str, np.ndarray], similarity_threshold: float
    ) -> np.ndarray:
        """Extracts matched pairs from similarity scores.

        Parameters
        ----------
        similarity_matrices : dict[str, ndarray(dtype=float64, ndim=2)]
            Dictionary with similarity matrices for each block.
        similarity_threshold : float
            Similarity threshold used for matching.

        Returns
        ----------
        matched_pairs : ndarray(dtype=int64, ndim=2)
            Array of matched pairs containing indices.
        """
        matched_pairs = []
        for block, matrix in similarity_scores.items():
            df_block = self.df[self.df["block"] == block]
            indices = np.triu(matrix >= similarity_threshold, k=1).nonzero()
            matched_pairs_block = np.hstack((indices[0][:, None], indices[1][:, None]))
            a = df_block.iloc[matched_pairs_block[:, 0]].index.to_numpy()
            b = df_block.iloc[matched_pairs_block[:, 1]].index.to_numpy()
            matched_pairs.append(np.hstack((a[:, np.newaxis], b[:, np.newaxis])))
        return np.concatenate(matched_pairs, axis=0)

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
        for attr, f in self.matching_fns.items():
            logging.info(f"Attribute '{attr}' is matched using function {f.__name__}")
        similarity_scores = self._get_similarity_matrices()
        self.matched_pairs = self._get_matched_pairs(
            similarity_scores, similarity_threshold
        )
        pipeline_execution_time = time.time() - pipeline_start_time
        logging.info(f"Writing the matched paper_ids to directory {dir_name}")
        self._write_matched_entities_csv(
            self.matched_pairs, dir_name, similarity_threshold
        )
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
            clusters = Pipeline._cluster_matched_entities(self.matched_pairs)
            logging.info(f"Writing the resolved dataset to directory {dir_name}")
            self._write_resolved_data(clusters, dir_name)
