import re

import dask.dataframe as dd


def all_lowercase_and_stripped_dask(df: dd.DataFrame) -> dd.DataFrame:
   
    df = df.map_partitions(lambda x: x.lower())  # all lowercase
    df["paper_title"] = df["paper_title"].map(
        lambda x: x.rstrip(".")
    )  # remove trailing '.'
    df["author_names"] = df["author_names"].map(
        lambda x: re.sub(r"\d", "", x)
    )  # remove digits
    return df
