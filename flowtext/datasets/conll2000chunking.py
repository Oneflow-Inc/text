from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract,
    _create_dataset_directory,
    _create_data_from_iob,
)
import os
import logging

URL = {
    "train": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/CoNLL2000Chunking/train.txt.gz",
    "test": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/CoNLL2000Chunking/test.txt.gz",
}

NUM_LINES = {
    "train": 8936,
    "test": 2012,
}

_EXTRACTED_FILES = {"train": "train.txt", "test": "test.txt"}

DATASET_NAME = "CoNLL2000Chunking"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def CoNLL2000Chunking(root, split):
    # Create a dataset specific subfolder to deal with generic download filenames
    root = os.path.join(root, "conll2000chunking")
    path = os.path.join(root, split + ".txt.gz")
    data_filename = _download_extract(
        root, URL[split], path, os.path.join(root, _EXTRACTED_FILES[split])
    )
    logging.info("Creating {} data".format(split))
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_iob(data_filename, " ")
    )
