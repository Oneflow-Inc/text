from functools import update_wrapper
from flowtext.utils import download_from_url, extract_archive
from flowtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _create_data_from_iob,
)

URL = "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/UDPOS/en-ud-v2.zip"

NUM_LINES = {
    "train": 12543,
    "valid": 2002,
    "test": 2077,
}


DATASET_NAME = "UDPOS"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def UDPOS(root, split):
    dataset_tar = download_from_url(URL, root=root)
    extracted_files = extract_archive(dataset_tar)
    if split == "valid":
        path = _find_match("dev.txt", extracted_files)
    else:
        path = _find_match(split + ".txt", extracted_files)
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_iob(path)
    )
