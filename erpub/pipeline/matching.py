import numpy as np
from scipy.spatial.distance import cosine


def jaccard_similarity(a: str, b: str) -> float:
    """
    Calculate the average Jaccard Similarity across a given attribute.
    """
    A, B = set(a.split()), set(b.split())
    intersection = A.intersection(B)
    union = A.union(B)
    score = len(intersection) / len(union)
    return score


def vector_embeddings(a: str, b: str, embedding_table: dict[str, np.ndarray]) -> float:
    """Use word embeddings to map tokens to vectors,
    average them and then return the cosine similarity"""
    mean_vec_a = np.mean([embedding_table[token] for token in a.split()], axis=0)
    mean_vec_b = np.mean([embedding_table[token] for token in b.split()], axis=0)
    cos_sim = 1 - cosine(mean_vec_a, mean_vec_b)
    return cos_sim
