from ..utils import download_from_url
from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os

URL = {
    "train": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/AG_NEWS/train.csv",
    "test": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/AG_NEWS/test.csv",
}

NUM_LINES = {
    "train": 120000,
    "test": 7600,
}

DATASET_NAME = "AG_NEWS"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=4)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AG_NEWS(root, split):
    path = download_from_url(
        URL[split], root=root, path=os.path.join(root, split + ".csv"),
    )
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_csv(path)
    )
