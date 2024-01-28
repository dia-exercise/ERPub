import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(a: str, b: str) -> float:
    """Calculate the average Jaccard Similarity across a given attribute."""
    A, B = set(a.split()), set(b.split())
    intersection = A.intersection(B)
    union = A.union(B)
    score = len(intersection) / len(union)
    return score


def vector_embeddings(
    data: pd.Series, embedding_table: dict[str, np.ndarray]
) -> np.ndarray:
    """Use word embeddings to map tokens to vectors,
    average them and then return the cosine similarity"""
    embedded_df = data.apply(
        lambda attr: np.mean([embedding_table[token] for token in attr.split()], axis=0)
    )
    return cosine_similarity(np.stack(embedded_df))
