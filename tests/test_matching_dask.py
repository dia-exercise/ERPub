import numpy as np
import pandas as pd
import pytest

from erpub.pipeline.matching_dask import (
    jaccard_similarity,
    specific_name_matcher_dask,
    vector_embeddings,
)


@pytest.fixture
def sample_series():
    return pd.Series(
        [
            "Foo Foo",
            "bar",
            "foo foo",
        ]
    )


@pytest.fixture
def sample_embedding_table():
    vec1 = np.random.rand(
        20,
    )
    vec2 = np.random.rand(
        20,
    )
    return {
        "foo": vec1,
        "bar": vec2,
        "Foo": vec1,
    }


def test_jaccard_similarity(sample_series):
    assert jaccard_similarity(sample_series, "baaa") == 0
    assert jaccard_similarity(sample_series, "foo") < 0.34


def test_vector_embeddings(sample_series, sample_embedding_table):
    similarity_matrix = vector_embeddings(sample_series, sample_embedding_table)
    assert similarity_matrix.shape == (3, 3)
    assert similarity_matrix[0, 2] > 0.999
    assert similarity_matrix[0, 1] < 1


def test_specific_name_matcher():
    assert specific_name_matcher_dask("J. M. Doe", "John Doe") == 1.0
    assert specific_name_matcher_dask("John Doe", "Jane Doe") == 1.0
    assert specific_name_matcher_dask("John Doe", "Bob Smith") == 0.0
