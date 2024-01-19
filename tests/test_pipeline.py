from collections import defaultdict
import csv
import os
from tempfile import TemporaryDirectory
import numpy as np

import pytest
from erpub.pipeline.matching import jaccard_similarity, vector_embeddings

from erpub.pipeline.pipeline import Pipeline


@pytest.fixture
def temp_embeddings_dir():
    "Creates a temp directory with a embeddings file"
    with TemporaryDirectory() as temp_dir:
        embeddings_dir = os.path.join(temp_dir, "sample_embeddings.txt")
        with open(embeddings_dir, "w") as f:
            f.write(
                "foo 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566\n"
            )
            f.write(
                "bar 0.013441 0.23682 -0.16899 0.40951 0.63812 0.47709 -0.42852 -0.55641 -0.364 -0.23938\n"
            )
        yield embeddings_dir


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

    assert pipeline.embedding_table is None


def test_pipeline_with_embedding_table(temp_csv_dir, temp_embeddings_dir):
    pipeline = Pipeline(
        temp_csv_dir,
        matching_fns={
            "paper_title": vector_embeddings,
            "author_names": jaccard_similarity,
        },
        embeddings_for_matching=temp_embeddings_dir,
    ).run()
    assert type(pipeline.embedding_table) is defaultdict


def test_pipeline_missing_embedding_table_error(temp_csv_dir):
    with pytest.raises(ValueError) as e:
        pipeline = Pipeline(
            temp_csv_dir,
            matching_fns={
                "paper_title": vector_embeddings,
                "author_names": jaccard_similarity,
            },
        ).run()


def test_get_embedding_table(temp_embeddings_dir):
    embeddings_table = Pipeline._get_embedding_table(temp_embeddings_dir)
    assert len(embeddings_table) == 2
    assert list(embeddings_table.keys()) == ["foo", "bar"]
    assert all(arr.shape == (10,) for arr in embeddings_table.values())
    assert type(embeddings_table["unknown"]) is np.ndarray


def test_requires_embedding_table():
    assert Pipeline._requires_embedding_table(lambda a, b, embedding_table: 1.0)
    assert not Pipeline._requires_embedding_table(lambda a, b: 1.0)
