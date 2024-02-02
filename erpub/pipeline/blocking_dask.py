import dask.dataframe as dd


def naive_all_pairs_dask(data: dd.DataFrame) -> None:
    """Create n*(n-1)/2 pairs to be compared."""
    data["block"] = 1


def same_year_of_publication_dask(data: dd.DataFrame) -> None:
    """Block based on the year of publication."""
    # ngroup() is not supported in dask :/
    data["block"] = data["year_of_publication"]


def author_names_initials_dask(data: dd.DataFrame) -> None:
    """Block based on the initials of the author names."""
    initials = data["author_names"].map_partitions(
        extract_initials_for_partition, meta=("author_names", "object")
    )
    data["block"] = initials


def extract_initials_for_partition(author_names: dd.Series) -> dd.Series:
    """A function to apply to each partition of the DataFrame, extracting initials."""

    def get_initials(name):
        name_list = name.split(",")
        name_list = [n.strip().split(" ") for n in name_list]
        initials = [
            "".join([part[0] for part in name_parts if part])
            for name_parts in name_list
        ]
        initials = [
            initials[i][0] + initials[i][-1] if len(initials[i]) > 1 else initials[i][0]
            for i in range(len(initials))
        ]
        return ",".join(sorted(initials))

    return author_names.apply(get_initials)
