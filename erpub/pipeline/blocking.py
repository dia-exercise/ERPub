from collections.abc import Sequence

import numpy as np


def naive_all_pairs(data: Sequence) -> np.ndarray:
    """Create n*(n-1)/2 pairs to be compared."""
    n = len(data)
    indices = np.triu_indices(n, k=1)
    tuples_array = np.column_stack(indices).astype(int)
    return tuples_array
