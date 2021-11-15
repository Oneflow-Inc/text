import logging
from flowtext.utils import download_from_url
from flowtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _read_text_iterator,
)

URL = {
    "train": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/PennTreebank/ptb.train.txt",
    "test": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/PennTreebank/ptb.test.txt",
    "valid": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/PennTreebank/ptb.valid.txt",
}

NUM_LINES = {
    "train": 42068,
    "valid": 3370,
    "test": 3761,
}

DATASET_NAME = "PennTreebank"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def PennTreebank(root, split):
    path = download_from_url(URL[split], root=root)
    logging.info("Creating {} data".format(split))
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _read_text_iterator(path)
    )
