import glob
import logging
import os
from typing import Callable, Sequence, Iterable

import pandas as pd

from erpub.pipeline.blocking import naive_all_pairs
from erpub.pipeline.clustering import connected_components_
from erpub.pipeline.matching import jaccard_similarity

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class Pipeline:
    def __init__(
        self,
        file_dir: str,
        preprocess_data_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        blocking_fn: Callable[[Sequence], Iterable[tuple[int]]] = naive_all_pairs,
        matching_fn: Callable[
            [Sequence[str], Sequence[str]], float
        ] = jaccard_similarity,
        clustering_fn: Callable[
            [Iterable[Sequence[int]], Iterable[float], int], Sequence[Iterable[int]]
        ] = connected_components_,
    ):
        self.df = Pipeline.load_data(file_dir)
        self.preprocess_data_fn = preprocess_data_fn
        self.blocking_fn = blocking_fn
        self.matching_fn = matching_fn
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

    def run(self):
        """
        Execute the entity resolution pipeline.
        """
        if self.preprocess_data_fn:
            self.df = self.preprocess_data_fn(self.df)
            logging.info("Data has been preprocessed")
        else:
            logging.info("Preprocessing has been skipped")
        self.pairs = self.blocking_fn(self.df)
        logging.info(
            f"Created list of pairs through blocking function {self.blocking_fn.__name__}"
        )
        self.sims = [
            self.matching_fn(self.df.iloc[a], self.df.iloc[b]) for a, b in self.pairs
        ]
        logging.info(
            f"Calculated similarities of pairs with matching function {self.matching_fn.__name__}"
        )
        self.clusters = self.clustering_fn(
            pairs=self.pairs, sims=self.sims, n=len(self.df)
        )
        logging.info(
            f"Deduplicated data into clusters clustering function {self.clustering_fn.__name__}"
        )
