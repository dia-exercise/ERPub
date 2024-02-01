import pandas as pd
import pytest

from erpub.pipeline.blocking import (
    author_names_initials,
    naive_all_pairs,
    same_year_of_publication,
)


@pytest.fixture
def sample_df():
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


def test_length_naive_all_pairs(sample_df):
    naive_all_pairs(sample_df)
    assert "block" in sample_df
    assert sample_df["block"].nunique() == 1


def test_same_year_of_publication(sample_df):
    same_year_of_publication(sample_df)
    assert "block" in sample_df
    assert sample_df["block"].nunique() == 3


def test_author_names_initials(sample_df):
    author_names_initials(sample_df)
    assert "block" in sample_df
    assert sample_df["block"].nunique() == 5

    values = sample_df["block"].values
    assert values[0] == values[1] == values[3]
    assert values[4] == values[6]
    assert values[5] == values[8] == values[9]
