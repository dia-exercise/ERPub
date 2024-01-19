from erpub.pipeline.blocking import naive_all_pairs, same_year_of_publication
import pytest
import pandas as pd


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


def test_length_naive_all_pairs():
    data = range(10)
    pairs_to_match = naive_all_pairs(data)
    assert list(pairs_to_match[0]) == [0, 1]
    assert len(pairs_to_match) == 45


def test_same_year_of_publication(sample_df):
    pairs_to_match = same_year_of_publication(sample_df)
    assert list(pairs_to_match[0]) == [0, 1]
    assert len(pairs_to_match) == 18
    assert (
        3 not in pairs_to_match.flatten()
    ), "Clusters with only 1 entry should be skipped"
