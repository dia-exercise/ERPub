import pandas as pd
import pytest

from erpub.pipeline.blocking import naive_all_pairs, same_year_of_publication


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
