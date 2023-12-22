from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connected_components_(
    pairs: Iterable[Sequence[int]],
    sims: Iterable[float],
    n: int,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """
    Clusters nodes contained in pairs based on the provided similarity.
    This is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html
    """
    graph = np.zeros((n, n))
    for (node1, node2), similarity in zip(pairs, sims):
        graph[node1, node2] = 1 if similarity > threshold else 0
        graph[node2, node1] = 1 if similarity > threshold else 0

    sparse_matrix = csr_matrix(graph)

    n_components, labels = connected_components(sparse_matrix, directed=False)
    components = [np.where(labels == i)[0] for i in range(n_components)]

    return components
