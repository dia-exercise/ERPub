import pytest

import numpy as np
import pandas as pd
from erpub.pipeline.matching import jaccard_similarity, vector_embeddings


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
    vec1 = np.random.rand(    20,)
    vec2 = np.random.rand(20,)
    return {
        "foo": vec1,
        "bar": vec2,
        "Foo": vec1,
    }


def test_jaccard_similarity():
    assert jaccard_similarity("Matt Böhm", "Matt Boehm") == 1 / 3
    assert jaccard_similarity("A", "A") == 1


def test_jaccard_similarity_fails_on_empty_attributes():
    with pytest.raises(ZeroDivisionError) as e:
        jaccard_similarity("", "")


def test_vector_embeddings(sample_series, sample_embedding_table):
    similarity_matrix = vector_embeddings(sample_series, sample_embedding_table)
    assert similarity_matrix.shape == (3,3)
    assert similarity_matrix[0,2] > 0.999
    assert similarity_matrix[0,1] < 1
