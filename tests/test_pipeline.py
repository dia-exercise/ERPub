import csv
import os
from tempfile import TemporaryDirectory
import numpy as np

import pytest

from erpub.pipeline.pipeline import Pipeline


@pytest.fixture
def temp_csv_dir():
    "Creates a temp directory with 2 csv files"
    columns = [
        "paper_id",
        "paper_title",
        "author_names",
        "publication_venue",
        "year_of_publication",
    ]

    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "acm_sample.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerow(
                [
                    "5390972920f70186a0dfac85",
                    "The next database revolution",
                    "Jim Gray",
                    "SIGMOD '04 Proceedings of the 2004 ACM SIGMOD international conference on Management of data",
                    "2004",
                ]
            )
            writer.writerow(
                [
                    "5390882d20f70186a0d8dad0",
                    "Efficient execution of joins in a star schema",
                    "Andreas Weininger",
                    "Proceedings of the 2002 ACM SIGMOD international conference on Management of data",
                    "2002",
                ]
            )

        with open(os.path.join(temp_dir, "dblp_sample.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerow(
                [
                    "53e9a515b7602d9702e350a0",
                    "An initial study of overheads of eddies.",
                    "Amol Deshpande",
                    "SIGMOD Record",
                    "2004",
                ]
            )
            writer.writerow(
                [
                    "53e99aecb7602d9702373cec",
                    "The next database revolution.",
                    "Jim Gray",
                    "SIGMOD Conference",
                    "2004",
                ]
            )
        yield temp_dir


def test_default_pipeline(temp_csv_dir):
    "Tests the dimensions and whether the given entities are resolved as expected"
    pipeline = Pipeline(temp_csv_dir).run()
    assert pipeline.pairs.shape == (6, 2)
    assert pipeline.sims.shape == (6,)
    assert len(pipeline.clusters) == 3

    expected_match = (
        pipeline.df.index[
            pipeline.df["paper_id"] == "53e99aecb7602d9702373cec"
        ].tolist()
        + pipeline.df.index[
            pipeline.df["paper_id"] == "5390972920f70186a0dfac85"
        ].tolist()
    )
    assert np.isclose(
        pipeline.sims[np.where(np.all(pipeline.pairs == expected_match, axis=1))],
        0.66923077,
    )
    assert expected_match in [list(elem) for elem in pipeline.clusters]
