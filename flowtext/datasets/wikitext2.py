import logging
from ..utils import download_from_url, extract_archive
from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _read_text_iterator,
)

URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"

NUM_LINES = {
    "train": 36718,
    "valid": 3760,
    "test": 4358,
}

DATASET_NAME = "WikiText2"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def WikiText2(root, split):
    dataset_tar = download_from_url(URL, root=root)
    extracted_files = extract_archive(dataset_tar)
    path = _find_match(split, extracted_files)
    logging.info("Creating {} data".format(split))
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _read_text_iterator(path)
    )
