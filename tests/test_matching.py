from erpub.pipeline.matching import jaccard_similarity


def test_jaccard_similarity():
    assert jaccard_similarity(("Matt Böhm",), ("Matt Boehm",)) == 1 / 3
    assert jaccard_similarity(("A",), ("A",)) == 1
    assert jaccard_similarity([], []) == 0
