import re

import pandas as pd


def all_lowercase_and_stripped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: x.lower())  # all lowercase
    df["paper_title"] = df["paper_title"].apply(
        lambda x: x.rstrip(".")
    )  # remove trailing '.'
    df["author_names"] = df["author_names"].apply(
        lambda x: re.sub(r"\d", "", x)
    )  # remove digits
    return df
