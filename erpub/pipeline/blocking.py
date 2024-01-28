import pandas as pd


def naive_all_pairs(data: pd.DataFrame) -> None:
    """Create n*(n-1)/2 pairs to be compared."""
    data["block"] = 1


def same_year_of_publication(data: pd.DataFrame) -> None:
    data["block"] = data.groupby("year_of_publication").ngroup()


def author_names_initials(data: pd.DataFrame) -> None:
    data["initials"] = data["author_names"]
    data["initials"] = data["initials"].apply(_extract_initials)
    data["block"] = data.groupby("initials").ngroup()
    data.drop("initials", axis=1, inplace=True)


def _extract_initials(names: str) -> str:
    # assumption: names are separated by a comma
    name_list = names.split(",")
    name_list = [n.strip() for n in name_list]
    # assumption: parts of names are separated by spaces
    name_list = [n.split(" ") for n in name_list]
    name_list = [[part.strip() for part in name_parts] for name_parts in name_list]
    # get initials
    name_list = [[part[0] for part in name_parts] for name_parts in name_list]
    # only retain the initials of the first name and the last name; ignore middle names
    name_list = [[initials[0], initials[-1]] for initials in name_list]
    # join initials into a single word per name
    name_list = ["".join(l) for l in name_list]
    # sort initials to account for different name orders
    name_list = sorted(name_list)
    # separate initials of different authors by comma (no space)
    return ",".join(name_list)
