import csv
import tarfile
import requests
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def download_input(url, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    target_path = f"{target_dir}/{url.split('/')[-1]}"
    if Path(target_path).is_file():
        logging.info(f"File {target_path} already exists. Won't download again")
        return

    logging.info(f"Start downloading {url}")
    # stream mode will first get the response header only
    response = requests.get(url, stream=True)
    with open(target_path, mode="wb") as file:
        # download file in chunks of 1MiB
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file.write(chunk)
    logging.info(f"File saved to {target_path}")

    logging.info(f"Extracting {target_path}")
    with tarfile.open(target_path, "r") as file:
        file.extractall(target_dir)


def set_dict_value(dict_: dict, key, value):
    if dict_.get(key):
        raise Exception(f"Key {key} already exists")
    else:
        dict_[key] = value

def read_txt(input_file: str) -> list[dict]:
    current_block = {}
    filtered_blocks = []

    logging.info(f"Start reading {input_file}")
    with open(input_file, "r") as file:
        # use set_dict_value() to populate dict. this is to ensure that the input file is properly formatted:
        # if a block contains two lines of the same type (paper references are ignored), an exception will be thrown
        # if two blocks aren't separated by a newline, an exception will be thrown
        for line in file:
            if line.startswith("#*"):
                set_dict_value(current_block, "paper_title", line[2:].rstrip())
            elif line.startswith("#@"):
                set_dict_value(current_block, "author_names", line[2:].rstrip())
            elif line.startswith("#t"):
                set_dict_value(current_block, "year_of_publication", line[2:].rstrip())
            elif line.startswith("#c"):
                set_dict_value(current_block, "publication_venue", line[2:].rstrip())
            elif line.startswith("#index"):
                set_dict_value(current_block, "paper_id", line[6:].rstrip())

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


def write_csv(blocks: list[dict], target_dir: str, output_file: str):
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    target_path = f"{target_dir}/{output_file}"
    logging.info(f"Write {target_path}")
    with open(target_path, "w") as file:
        fields = ["paper_id", "paper_title", "author_names", "publication_venue", "year_of_publication"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(blocks)


if __name__ == "__main__":
    dblp_url = "https://lfs.aminer.cn/lab-datasets/citation/dblp.v8.tgz"
    acm_url = "https://lfs.aminer.cn/lab-datasets/citation/citation-acm-v8.txt.tgz"
    # TODO where data is saved currently depends on the Python working directory
    #  i.e. calling the script from the root will result in ERPub/data
    #  calling it from erpub will result in ERPub/erpub/data
    #  we can decide for a path and make this consistent
    download_target_dir = "data/raw"
    prepared_target_dir = "data/prepared"

    download_input(dblp_url, download_target_dir)
    dblp_blocks = read_txt(f"{download_target_dir}/dblp.txt")
    write_csv(dblp_blocks, prepared_target_dir, "DBLP_1995_2004.csv")

    download_input(acm_url, download_target_dir)
    acm_blocks = read_txt(f"{download_target_dir}/citation-acm-v8.txt")
    write_csv(acm_blocks, prepared_target_dir, "ACM_1995_2004.csv")
