import dask.dataframe as dd


def all_lowercase_and_stripped_dask(df: dd.DataFrame) -> dd.DataFrame:
    df["paper_title"] = df["paper_title"].str.lower()
    df["author_names"] = df["author_names"].str.lower()
    df["publication_venue"] = df["publication_venue"].str.lower()
    df["paper_title"] = df["paper_title"].str.rstrip(".")
    df["author_names"] = df["author_names"].str.replace(r"\d", "")
    return df
