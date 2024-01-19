from collections.abc import Sequence

import numpy as np
import pandas as pd


def naive_all_pairs(data: Sequence) -> np.ndarray:
    """Create n*(n-1)/2 pairs to be compared."""
    n = len(data)
    indices = np.triu_indices(n, k=1)
    tuples_array = np.column_stack(indices).astype(int)
    return tuples_array


def same_year_of_publication(data: pd.DataFrame) -> np.ndarray:
    clusters = data.groupby("year_of_publication").groups.values()
    pairs_list = [
        np.column_stack((arr[i], arr[j]))
        for arr in clusters
        if len(arr) > 1  # Exclude clusters with only one entry
        for i, j in [np.triu_indices(len(arr), k=1)]
    ]
    return np.concatenate(pairs_list, axis=0)
