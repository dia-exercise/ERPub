import csv
import os.path
import tarfile
import requests
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def download_input(url, target_dir):
    """
    Download tarball from url in streaming mode and extract it in target_dir.
    Skip download if tarball already exists in target_dir.
    """
    abs_target_dir = _get_abs_path(target_dir)
    Path(abs_target_dir).mkdir(parents=True, exist_ok=True)

    target_path = f"{abs_target_dir}/{url.split('/')[-1]}"
    if Path(target_path).is_file():
        logging.info(f"File {target_path} already exists. Won't download again")
        return

    logging.info(f"Start downloading {url}")
    # streaming mode will first get the response header only
    response = requests.get(url, stream=True)
    with open(target_path, mode="wb") as file:
        # download file in chunks of 1MiB
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file.write(chunk)
    logging.info(f"File saved to {target_path}")

    logging.info(f"Extracting {target_path}")
    with tarfile.open(target_path, "r") as file:
        file.extractall(target_dir)


def read_txt(input_file: str) -> list[dict]:
    """
    Read in txt file, formatted as described here: https://www.aminer.org/citation (v8).
    Extract publication blocks. Filter out publications published before 1995 or after 2004. Filter out publications that
    don't have 'sigmod' or 'vldb' in their publication venue name.
    """
    abs_input_path = _get_abs_path(input_file)

    current_block: dict[str, str] = {}
    filtered_blocks = []

    logging.info(f"Reading {abs_input_path}")
    with open(abs_input_path, "r") as file:
        # use set_dict_value() to populate dict. this is to ensure that the input file is properly formatted:
        # if a block contains two lines of the same type (except paper references), an exception will be thrown
        # if two blocks aren't separated by a newline, an exception will be thrown
        for line in file:
            if line.startswith("#*"):
                _set_dict_value(current_block, "paper_title", line[2:].rstrip())
            elif line.startswith("#@"):
                _set_dict_value(current_block, "author_names", line[2:].rstrip())
            elif line.startswith("#t"):
                _set_dict_value(current_block, "year_of_publication", line[2:].rstrip())
            elif line.startswith("#c"):
                _set_dict_value(current_block, "publication_venue", line[2:].rstrip())
            elif line.startswith("#index"):
                _set_dict_value(current_block, "paper_id", line[6:].rstrip())

            elif line == "\n":
                # exclude publications that don't provide the attributes we filter on
                if not current_block.get("publication_venue") or not current_block.get("year_of_publication"):
                    current_block = {}
                    continue

                pub_venue_lower = current_block["publication_venue"].lower()
                year = int(current_block["year_of_publication"])
                if ("sigmod" in pub_venue_lower or "vldb" in pub_venue_lower) and 1995 <= year <= 2004:
                    filtered_blocks.append(current_block)

                current_block = {}

    logging.info(f"Found {len(filtered_blocks)} publications matching criteria")
    return filtered_blocks


def remove_duplicates(blocks: list[dict]) -> list[dict]:
    """Remove duplicated publication blocks - i.e. all dict values are the same."""

    df = pd.DataFrame(blocks)
    df_dedup = df.drop_duplicates()
    logging.info(f"Removed {df.shape[0] - df_dedup.shape[0]} duplicate publication entries")
    return df_dedup.to_dict("records")


def write_csv(blocks: list[dict], target_dir: str, output_file: str) -> None:
    """Write publication blocks into csv file in target_dir."""

    abs_target_dir = _get_abs_path(target_dir)
    Path(abs_target_dir).mkdir(parents=True, exist_ok=True)

    target_path = os.path.join(abs_target_dir, output_file)
    logging.info(f"Writing {target_path}")
    with open(target_path, "w") as file:
        fields = ["paper_id", "paper_title", "author_names", "publication_venue", "year_of_publication"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(blocks)


def _get_abs_path(rel_path: str) -> str:
    """Turn rel_path into an absolute path, relative to this file."""
    abs_file_dir = os.path.dirname(__file__)
    return os.path.join(abs_file_dir, rel_path)


def _set_dict_value(dict_: dict, key: str, value: str) -> None:
    """Set key to value in dict if key does not yet exist. If it does already exist, throw exception."""
    if dict_.get(key):
        raise Exception(f"Key {key} already exists")
    else:
        dict_[key] = value


if __name__ == "__main__":
    dblp_url = "https://lfs.aminer.cn/lab-datasets/citation/dblp.v8.tgz"
    acm_url = "https://lfs.aminer.cn/lab-datasets/citation/citation-acm-v8.txt.tgz"
    download_target_dir = "data/raw"
    prepared_target_dir = "data/prepared"

    download_input(dblp_url, download_target_dir)
    dblp_blocks = read_txt(f"{download_target_dir}/dblp.txt")
    dblp_blocks = remove_duplicates(dblp_blocks)
    write_csv(dblp_blocks, prepared_target_dir, "DBLP_1995_2004.csv")

    download_input(acm_url, download_target_dir)
    acm_blocks = read_txt(f"{download_target_dir}/citation-acm-v8.txt")
    acm_blocks = remove_duplicates(acm_blocks)
    write_csv(acm_blocks, prepared_target_dir, "ACM_1995_2004.csv")
