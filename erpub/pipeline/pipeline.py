import glob
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from statistics import fmean

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
        match_dir: str,
        resolved_files_dir: str,
        similarity_threshold: float,
        preprocess_data_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        blocking_fn: Callable[[Sequence], Iterable[tuple[int, int]]] = naive_all_pairs,
        matching_fns: dict[str, Callable[[str, str], float]] = {
            attr: jaccard_similarity for attr in DEFAULT_ATTRIBUTES
        },
        cluster_matched_entities : bool = False
    ):
        self.df = Pipeline.load_data(file_dir)
        self.match_dir = match_dir
        self.resolved_files_dir = resolved_files_dir
        self.similarity_threshold = similarity_threshold
        self.preprocess_data_fn = preprocess_data_fn
        self.blocking_fn = blocking_fn
        self.matching_fns = matching_fns
        self.cluster_matched_entities = cluster_matched_entities
        self.pairs_to_match = None
        self.similarity_scores = None
        self.matched_df = None
        logging.info("Pipeline initialized")

    @staticmethod
    def load_data(dir: str) -> pd.DataFrame:
        """Load all csv files in the specified directory into a single pandas DataFrame."""
        all_files = glob.glob(os.path.join(dir, "*.csv"))
        logging.info(f"The pipeline will be built with these files: {all_files}")
        df = pd.concat((pd.read_csv(f).assign(dataset=f) for f in all_files), ignore_index=True).astype(str)
        logging.info("Loaded csv successfully into pandas dataframe")
        return df
    
    def _write_matched_entities_csv(self, matched_pairs: Iterable[tuple[int, int]]) -> None:
        """Writes for each match between entities from different datasets an entry to the match_dir."""
        unique_datasets = self.df.dataset.unique()
        match_ids: list[dict[str, str]] = [self.df.iloc[match].set_index('dataset').paper_id.to_dict() for match in matched_pairs if len(self.df.iloc[match].dataset.unique()) == len(unique_datasets)]
        if self.match_dir:
            run_dir = self._get_run_directory()
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(match_ids).to_csv(os.path.join(run_dir, "matched_entities.csv"), index=False)
            with open(os.path.join(run_dir, 'pipeline_settings.txt'), 'w') as file:
                file.write(f"Blocking function: {self.blocking_fn.__name__}\n")
                for attr, f in self.matching_fns.items():
                    file.write(f"Matching function for attribute {attr}: {f.__name__}\n")
                file.write(f"Similarity threshold: {self.similarity_threshold}\n")
    
    def _get_run_directory(self) -> str:
        """Identifies what was the last run of the pipeline and returns the new dir name of the current run."""
        existing_runs = [d for d in os.listdir(self.match_dir) if d.isdigit()]
        next_run_number = str(max(existing_runs) + 1) if existing_runs else "0"
        return os.path.join(self.match_dir, next_run_number)

    def _cluster_matched_entities(self, matched_pairs: Iterable[tuple[int, int]]) -> list[set[int]]:
        """Clusters the matched_pairs."""
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
    
    def _write_resolved_data(self, clusters: list[set[int]]):
        """Based on the clusters removes all found duplicates from all datasets and writes the updated csv to the resolved_files_dir."""
        df = self.df.copy()
        all_columns = DEFAULT_ATTRIBUTES + ["paper_id"]
        for cluster in clusters:
            cluster_list = list(cluster)
            df.loc[cluster_list, all_columns] = df.loc[cluster_list[0], all_columns].values
        df = df.drop_duplicates()
        Path(self.resolved_files_dir).mkdir(parents=True, exist_ok=True)
        for dataset in df.dataset.unique():
            dataset_df = df[df["dataset"] == dataset][all_columns]
            dataset_df.to_csv(os.path.join(self.resolved_files_dir, "resolved_" + dataset.split("/")[-1]), index=False)

    def run(self):
        """Execute the entity resolution pipeline."""
        if self.preprocess_data_fn:
            logging.info("Data will be preprocessed")
            self.df = self.preprocess_data_fn(self.df)
        else:
            logging.info("Preprocessing has been skipped")

        logging.info(
            f"Create list of pairs through blocking function {self.blocking_fn.__name__}"
        )
        self.pairs_to_match = self.blocking_fn(self.df)
        logging.info(f"Calculate similarities of {len(self.pairs_to_match)} pairs")
        for attr, f in self.matching_fns.items():
            logging.info(f"Attribute '{attr}' is matched using function {f.__name__}")
        self.similarity_scores = np.array(
            [
                fmean(
                    [
                        match_f(self.df.iloc[a][attr], self.df.iloc[b][attr])
                        for attr, match_f in self.matching_fns.items()
                    ]
                )
                for a, b in self.pairs_to_match
            ]
        )
        matched_pairs = self.pairs_to_match[self.similarity_scores > self.similarity_threshold]
        logging.info("Writing the matched paper_ids to directory %s", self.match_dir)
        self._write_matched_entities_csv(matched_pairs)

        if self.cluster_matched_entities:
            logging.info("Clustering matched entities")
            clusters = self._cluster_matched_entities(matched_pairs)
            logging.info("Writing the resolved dataset to directory %s", self.resolved_files_dir)
            self._write_resolved_data(clusters)
