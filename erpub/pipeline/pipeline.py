import glob
import logging
import os
import inspect
from collections.abc import Callable, Iterable, Sequence
from statistics import fmean
from collections import defaultdict

import numpy as np
import pandas as pd

from erpub.pipeline.blocking import naive_all_pairs
from erpub.pipeline.clustering import connected_components_
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
        blocking_fn: Callable[[Sequence], Iterable[tuple[int]]] = naive_all_pairs,
        matching_fns: dict[
            str,
            Callable[[str, str], float]
            | Callable[[str, str, dict[str, np.ndarray]], float],
        ] = {attr: jaccard_similarity for attr in DEFAULT_ATTRIBUTES},
        clustering_fn: Callable[
            [Iterable[Sequence[int]], Iterable[float], int], Sequence[Iterable[int]]
        ] = connected_components_,
        embeddings_for_matching: str | None = None,
    ):
        self.df = Pipeline.load_data(file_dir)
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
        self.clustering_fn = clustering_fn
        logging.info("Pipeline initialized")

    @staticmethod
    def load_data(dir: str) -> pd.DataFrame:
        """
        Load all csv files in the specified directory into a single pandas DataFrame.
        """
        all_files = glob.glob(os.path.join(dir, "*.csv"))
        logging.info(f"The pipeline will be built with these files: {all_files}")
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True).astype(
            str
        )
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
        return "embedding_table" in inspect.signature(match_f).parameters

    def run(self):
        """
        Execute the entity resolution pipeline.
        """
        if self.preprocess_data_fn:
            logging.info("Data will be preprocessed")
            self.df = self.preprocess_data_fn(self.df)
        else:
            logging.info("Preprocessing has been skipped")

        logging.info(
            f"Create list of pairs through blocking function {self.blocking_fn.__name__}"
        )
        self.pairs = self.blocking_fn(self.df)

        logging.info("Calculate similarities of pairs")
        for attr, f in self.matching_fns.items():
            logging.info(f"Attribute '{attr}' is matched using function {f.__name__}")
        self.sims = np.array(
            [
                fmean(
                    [
                        match_f(
                            self.df.iloc[a][attr],
                            self.df.iloc[b][attr],
                            self.embedding_table,
                        )
                        if Pipeline._requires_embedding_table(match_f)
                        else match_f(self.df.iloc[a][attr], self.df.iloc[b][attr])
                        for attr, match_f in self.matching_fns.items()
                    ]
                )
                for a, b in self.pairs
            ]
        )

        logging.info(
            f"Deduplicate data into clusters using clustering function {self.clustering_fn.__name__}"
        )
        self.clusters = self.clustering_fn(
            pairs=self.pairs, sims=self.sims, n=len(self.df)
        )

        return self
