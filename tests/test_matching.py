import pytest

import numpy as np

from erpub.pipeline.matching import jaccard_similarity, vector_embeddings


def test_jaccard_similarity():
    assert jaccard_similarity("Matt Böhm", "Matt Boehm") == 1 / 3
    assert jaccard_similarity("A", "A") == 1


def test_jaccard_similarity_fails_on_empty_attributes():
    with pytest.raises(ZeroDivisionError) as e:
        jaccard_similarity("", "")


def test_vector_embeddings():
    vec1, vec2 = np.random.rand(20,), np.random.rand(
        20,
    )
    embedding_table = {
        "foo": vec1,
        "bar": vec2,
        "Foo": vec1,
    }
    assert vector_embeddings("foo", "foo", embedding_table) == 1
    assert vector_embeddings("foo", "Foo", embedding_table) == 1
    assert vector_embeddings("Foo foo", "Foo bar", embedding_table) < 1
