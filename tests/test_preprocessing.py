import pandas as pd
import pytest

from erpub.pipeline.preprocessing import all_lowercase_and_stripped


@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {
        "paper_title": ["Title One.", "Title Two.", "Title Three."],
        "author_names": ["John Doe1", "Jane Doe2", "Bob Smith3"],
    }
    return pd.DataFrame(data)


def test_all_lowercase_and_stripped(sample_dataframe):
    # Call the method on the sample DataFrame
    result_df = all_lowercase_and_stripped(sample_dataframe.copy())

    # Check if the paper_title is all lowercase
    assert all(result_df["paper_title"].str.islower())

    # Check if trailing '.' is removed from paper_title
    assert all(result_df["paper_title"].str.endswith(".") == False)

    # Check if digits are removed from author_names
    assert all(result_df["author_names"].str.isnumeric() == False)
