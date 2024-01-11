import glob
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from statistics import fmean

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
        matching_fns: dict[str, Callable[[str, str], float]] = {
            attr: jaccard_similarity for attr in DEFAULT_ATTRIBUTES
        },
        clustering_fn: Callable[
            [Iterable[Sequence[int]], Iterable[float], int], Sequence[Iterable[int]]
        ] = connected_components_,
    ):
        self.df = Pipeline.load_data(file_dir)
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
                        match_f(self.df.iloc[a][attr], self.df.iloc[b][attr])
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
