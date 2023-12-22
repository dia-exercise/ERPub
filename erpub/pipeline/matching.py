from typing import Sequence


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Calculate the average Jaccard Similarity across multiple attributes.
    """
    score = 0.0
    for a_attr, b_attr in zip(a, b):
        A, B = set(a_attr.split()), set(b_attr.split())
        intersection = A.intersection(B)
        union = A.union(B)
        score += len(intersection) / len(union)
    return score / len(a)
