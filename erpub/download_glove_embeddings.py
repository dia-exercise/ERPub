import argparse
import logging
import os
import zipfile

import requests

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def download_and_extract_zip(url: str, destination_folder: str) -> None:
    """Download a zip file from the given URL and extract its contents to the destination folder.

    Parameters
    ----------
    url: str
        The URL of the zip file to download.
    destination_folder: str
        The destination folder to extract the contents.
    """
    os.makedirs(destination_folder, exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        logging.info("Downloaded the zip-file")
        zip_file_path = os.path.join(destination_folder, "downloaded_file.zip")
        with open(zip_file_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)

        os.remove(zip_file_path)
        logging.info(f"Extracted contents to {destination_folder}")
    else:
        logging.error(
            f"Failed to download the file. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a zip file and extract its contents."
    )
    parser.add_argument(
        "--url",
        help="URL of the zip file to download",
        default="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    )
    parser.add_argument(
        "--destination",
        help="Destination folder to extract the contents",
        default="embeddings",
    )
    args = parser.parse_args()

    download_and_extract_zip(args.url, args.destination)
