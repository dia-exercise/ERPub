import pandas as pd


def naive_all_pairs(data: pd.DataFrame) -> None:
    """Create n*(n-1)/2 pairs to be compared."""
    data["block"] = 1


def same_year_of_publication(data: pd.DataFrame) -> None:
    data["block"] = data.groupby("year_of_publication").ngroup()
