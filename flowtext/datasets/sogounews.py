from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os
import logging

URL = "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SogouNews/sogou_news_csv.tar.gz"

NUM_LINES = {
    "train": 450000,
    "test": 60000,
}

_PATH = "sogou_news_csv.tar.gz"

_EXTRACTED_FILES = {
    "train": f"{os.sep}".join(["sogou_news_csv", "train.csv"]),
    "test": f"{os.sep}".join(["sogou_news_csv", "test.csv"]),
}

DATASET_NAME = "SogouNews"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def SogouNews(root, split):
    path = _download_extract(
        root,
        URL,
        os.path.join(root, _PATH),
        os.path.join(root, _EXTRACTED_FILES[split]),
    )
    logging.info("Creating {} data".format(split))
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_csv(path)
    )
