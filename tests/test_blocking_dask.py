import pandas as pd
import dask.dataframe as dd
import pytest

from erpub.pipeline.blocking_dask import (
    naive_all_pairs_dask,
    same_year_of_publication_dask,
    author_names_initials_dask,
)


@pytest.fixture
def sample_pddf():
    return pd.DataFrame(
        {
            "year_of_publication": [
                1995,
                1995,
                1995,
                1998,
                2002,
                2002,
                2002,
                2002,
                2002,
                2002,
            ],
            "author_names": [
                "Kenneth A. Ross",
                "Kenneth A. Ross",
                "Huai Yang, Li Xu, Wynne Hsu",
                "Kenneth A. Ross",
                "David W. Embley, Li Xu, Yihong Ding",
                "C. J. Date",
                "Yihong Ding, David W. Embley, Li Xu",
                "Stephen Blott, Henry F.Korth",
                "C. J. Date",
                "C. J. Date",
            ],
            "other_column": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )


@pytest.fixture
def sample_df():
    return dd.from_pandas(
        pd.DataFrame(
            {
                "year_of_publication": [
                    1995,
                    1995,
                    1995,
                    1998,
                    2002,
                    2002,
                    2002,
                    2002,
                    2002,
                    2002,
                ],
                "author_names": [
                    "Kenneth A. Ross",
                    "Kenneth A. Ross",
                    "Huai Yang, Li Xu, Wynne Hsu",
                    "Kenneth A. Ross",
                    "David W. Embley, Li Xu, Yihong Ding",
                    "C. J. Date",
                    "Yihong Ding, David W. Embley, Li Xu",
                    "Stephen Blott, Henry F.Korth",
                    "C. J. Date",
                    "C. J. Date",
                ],
                "other_column": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        ),
        npartitions=3,
    )


def test_length_naive_all_pairs(sample_df):
    naive_all_pairs_dask(sample_df)
    assert "block" in sample_df.columns
    assert sample_df["block"].compute().nunique() == 1


def test_same_year_of_publication(sample_df):
    same_year_of_publication_dask(sample_df)
    assert "block" in sample_df.columns
    assert sample_df["block"].compute().nunique() == 3
    assert (
        sample_df["block"].compute().all()
        == sample_df["year_of_publication"].compute().all()
    )
    assert (
        sample_df["block"].compute().all()
        == dd.from_pandas(pd.Series([1995, 1998, 2002], name="block"), npartitions=1)
        .compute()
        .all()
    )


def test_author_names_initials(sample_df):
    author_names_initials_dask(sample_df)
    assert "block" in sample_df
    assert sample_df["block"].compute().nunique() == 5

    values = sample_df["block"].compute().values
    assert values[0] == values[1] == values[3]
    assert values[4] == values[6]
    assert values[5] == values[8] == values[9]
