import csv
import os
from collections import defaultdict
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from erpub.pipeline.blocking import naive_all_pairs
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

        with open(
            os.path.join(temp_dir, "dblp_sample.csv"), "w", newline="", encoding="utf-8"
        ) as f:
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


def test_default_pipeline(temp_csv_dir, mocker):
    "Tests the dimensions and whether the matched_entities is created"
    pipeline = Pipeline(file_dir=temp_csv_dir)
    mocker.patch.object(pipeline, "_write_matched_entities_csv")
    pipeline.run("blub", similarity_threshold=0.6)

    assert np.allclose(pipeline.matched_pairs, np.array([[1, 2]]))
    assert pipeline.embedding_table is None


def test_load_data(temp_csv_dir):
    df = Pipeline._load_data(temp_csv_dir)
    assert list(df.to_dict().keys()) == [
        "paper_id",
        "paper_title",
        "author_names",
        "publication_venue",
        "year_of_publication",
        "dataset",
    ]
    assert len(df) == 4


def test_write_matched_entities_csv(temp_csv_dir):
    pipeline = Pipeline(
        file_dir=temp_csv_dir,
        blocking_fn=naive_all_pairs,
        matching_fns={
            attr: jaccard_similarity
            for attr in [
                "paper_title",
                "author_names",
                "publication_venue",
                "year_of_publication",
            ]
        },
    )
    matched_pairs = np.array([[0, 3]])
    run_dir = os.path.join(temp_csv_dir, "my_dir")
    pipeline._write_matched_entities_csv(
        matched_pairs,
        run_dir,
        0.6,
    )

    run_dir_contents = os.listdir(run_dir)
    assert (
        "matched_entities.csv" in run_dir_contents
        and "pipeline_settings.txt" in run_dir_contents
    )

    matched_entities = pd.read_csv(
        os.path.join(run_dir, "matched_entities.csv")
    ).to_dict(orient="list")
    colums_end = [col.split("/")[-1] for col in matched_entities.keys()]
    values = [val[0] for val in matched_entities.values()]

    assert "dblp_sample.csv" in colums_end and "acm_sample.csv" in colums_end
    assert "53e9a515b7602d9702e350a0" in values and "5390882d20f70186a0d8dad0" in values


def test_pipeline_with_embedding_table(temp_csv_dir, temp_embeddings_dir):
    pipeline = Pipeline(
        temp_csv_dir,
        matching_fns={
            "paper_title": vector_embeddings,
            "author_names": jaccard_similarity,
        },
        embeddings_for_matching=temp_embeddings_dir,
    )
    pipeline.run(os.path.join(temp_csv_dir, "my_run"), 0.5)
    assert type(pipeline.embedding_table) is defaultdict


def test_pipeline_missing_embedding_table_error(temp_csv_dir):
    with pytest.raises(ValueError):
        Pipeline(
            temp_csv_dir,
            matching_fns={
                "paper_title": vector_embeddings,
                "author_names": jaccard_similarity,
            },
        )


def test_get_embedding_table(temp_embeddings_dir):
    embeddings_table = Pipeline._get_embedding_table(temp_embeddings_dir)
    assert len(embeddings_table) == 2
    assert list(embeddings_table.keys()) == ["foo", "bar"]
    assert all(arr.shape == (10,) for arr in embeddings_table.values())
    assert type(embeddings_table["unknown"]) is np.ndarray


def test_requires_embedding_table():
    assert Pipeline._requires_embedding_table(lambda a, b, embedding_table: 1.0)
    assert not Pipeline._requires_embedding_table(lambda a, b: 1.0)


def test_cluster_matched_entities():
    matched_pairs = np.array([[0, 1], [4, 6], [1, 7]])
    clusters = Pipeline._cluster_matched_entities(matched_pairs)
    assert len(clusters) == 2
    assert clusters[0] == {0, 1, 7}
    assert clusters[1] == {4, 6}


def test_pipeline_resolve_without_run(temp_csv_dir, mocker):
    pipeline = Pipeline(file_dir=temp_csv_dir)
    mocker.patch.object(Pipeline, "_cluster_matched_entities")
    mocker.patch.object(pipeline, "_write_resolved_data")

    resolve_dir = os.path.join(temp_csv_dir, "resolved_data")
    pipeline.resolve(resolve_dir)

    Pipeline._cluster_matched_entities.assert_not_called()
    pipeline._write_resolved_data.assert_not_called()


def test_pipeline_resolve(temp_csv_dir, mocker):
    pipeline = Pipeline(file_dir=temp_csv_dir)

    mocker.patch.object(Pipeline, "_cluster_matched_entities")
    mocker.patch.object(pipeline, "_write_resolved_data")

    run_dir = os.path.join(temp_csv_dir, "my_run")
    pipeline.run(run_dir, similarity_threshold=0.5)
    resolve_dir = os.path.join(temp_csv_dir, "resolved_data")
    pipeline.resolve(resolve_dir)

    Pipeline._cluster_matched_entities.assert_called_once()
    pipeline._write_resolved_data.assert_called_once()


def test_write_resolved_data(temp_csv_dir):
    pipeline = Pipeline(file_dir=temp_csv_dir)
    pipeline.original_df = pd.concat([pipeline.original_df, pipeline.original_df], ignore_index=True)
    clusters = [{0, 3, 5, 7}]
    resolved_dir = os.path.join(temp_csv_dir, "my_dir")
    pipeline._write_resolved_data(clusters, resolved_dir)
    assert ["resolved_acm_sample.csv", "resolved_dblp_sample.csv"] == sorted(
        os.listdir(resolved_dir)
    )
    acm_df = pd.read_csv(os.path.join(resolved_dir, "resolved_acm_sample.csv"))
    dblp_df = pd.read_csv(os.path.join(resolved_dir, "resolved_dblp_sample.csv"))
    expected_columns = [
        "paper_id",
        "paper_title",
        "author_names",
        "publication_venue",
        "year_of_publication",
    ]
    assert list(acm_df.to_dict().keys()) == expected_columns
    assert len(acm_df) == 2
    assert list(dblp_df.to_dict().keys()) == expected_columns
    assert len(dblp_df) == 2

    assert (
        acm_df["paper_id"][1] == dblp_df["paper_id"][0]
        or acm_df["paper_id"][1] == dblp_df["paper_id"][1]
    )


def test__get_similarity_matrices(mocker):
    mocker.patch.object(Pipeline, "_load_data")
    pipeline = Pipeline("foo")

    pipeline.df = pd.DataFrame(
        {
            "paper_title": ["this is foo", "this is foo", "something else"],
            "author_names": ["Mr Foo", "Mr Bar", "Someone else"],
            "block": ["1", "1", "1"],
        }
    )
    pipeline.matching_fns = {
        attr: jaccard_similarity for attr in ["paper_title", "author_names"]
    }

    similarity_matrices = pipeline._get_similarity_matrices()
    assert len(similarity_matrices) == 1
    assert similarity_matrices["1"].shape == (3, 3)
    assert np.all(similarity_matrices["1"] >= 0)
    assert similarity_matrices["1"][0, 1] > 0.6
