import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(df: dd.DataFrame, col: str) -> float:
    """Calculate the average Jaccard Similarity for a given attribute."""

    def compute_jaccard(b):
        A, B = set(col.split()), set(b.split())
        intersection = A.intersection(B)
        union = A.union(B)
        return len(intersection) / len(union) if union else 0

    return df.apply(compute_jaccard, meta=float, axis=1).mean()


def vector_embeddings(
    data: pd.Series, embedding_table: dict[str, np.ndarray]
) -> np.ndarray:
    """Use word embeddings to map tokens to vectors,
    average them and then return the cosine similarity"""
    embedded_df = data.apply(
        lambda attr: np.mean([embedding_table[token] for token in attr.split()], axis=0)
    )
    return cosine_similarity(np.stack(embedded_df))


def _name_to_initials_and_last_name(name: str) -> str:
    names = name.split()
    first_name_initial = names[0][0]
    last_name = names[-1]
    return f"{first_name_initial} {last_name}"


def specific_name_matcher_dask(a: str, b: str) -> float:
    names_a = set(
        _name_to_initials_and_last_name(name.strip()) for name in a.split(",")
    )
    names_b = set(
        _name_to_initials_and_last_name(name.strip()) for name in b.split(",")
    )
    return len(names_a & names_b) / max(len(names_a), len(names_b))
