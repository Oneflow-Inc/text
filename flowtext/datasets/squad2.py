from flowtext.utils import download_from_url
from flowtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_json,
)

URL = {
    "train": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SQuAD2/train-v2.0.json",
    "dev": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SQuAD2/dev-v2.0.json",
}

NUM_LINES = {
    "train": 130319,
    "dev": 11873,
}


DATASET_NAME = "SQuAD2"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev"))
def SQuAD2(root, split):
    extracted_files = download_from_url(URL[split], root=root)
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_json(extracted_files)
    )
