from erpub.pipeline.pipeline import Pipeline
from erpub.pipeline.preprocessing_dask import all_lowercase_and_stripped_dask
from erpub.pipeline.blocking_dask import author_names_initials_dask
from erpub.pipeline.matching import jaccard_similarity
import dask.dataframe as dd
import pandas as pd
import numpy as np

matching_fns = {"paper_title": np.vectorize(jaccard_similarity)}

def vectorize_simple_function(
    data, fn
):
    """Vectorizes a simple function over the elements of a pandas Series.

    Parameters
    ----------
    data : pd.Series
        The pandas Series containing the data.
    fn : Callable[[str, str], float]
        The function to be vectorized.

    Returns
    ----------
    similarity_matrix : ndarray(dtype=float64, ndim=2)
        Resulting matrix after applying the function to all pairs of elements in the Series.
        It's of size NxN where N is the length of the Series.
    """
    n = len(data)
    similarity_matrix = np.zeros((n, n))
    indices_a, indices_b = np.triu_indices(n)
    a = data.iloc[indices_a]
    b = data.iloc[indices_b]
    print(data)
    similarity_matrix[indices_a, indices_b] = fn(a, b)
    return similarity_matrix

def get_similarity_matrices(df):
    """Calculates a dict with the average similarity matrix over matching_fns for each block.
    Returns
    ----------
    similarity_matrices : dict[str, ndarray(dtype=float64, ndim=2)]
        Dict with similarity matrices for each cluster of size NxN where N is the elements in the block.
    """
    matrix_lst = [
        (
            vectorize_simple_function(df[attr], match_f)  # type: ignore
        )
        for attr, match_f in matching_fns.items()
    ]
    return matrix_lst.mean(0) # Average of the matching functions


if __name__ == '__main__':
    data = all_lowercase_and_stripped_dask(dd.from_pandas(Pipeline._load_data("data/prepared_small"), npartitions=100))
    author_names_initials_dask(data)
    blocks = data.block.unique().compute()
    data = data.set_index("block")
    data = data.repartition(divisions=sorted(blocks))
    # similarity_threshold = 0.8
    similarity_scores = data.map_partitions(get_similarity_matrices)
    # for p in data.partitions:
    #     print(p.paper_title.compute())
    print(similarity_scores.compute())
    print("done") 
