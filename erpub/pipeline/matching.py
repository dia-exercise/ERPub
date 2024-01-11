def jaccard_similarity(a: str, b: str) -> float:
    """
    Calculate the average Jaccard Similarity across a given attribute.
    """
    A, B = set(a.split()), set(b.split())
    intersection = A.intersection(B)
    union = A.union(B)
    score = len(intersection) / len(union)
    return score
