from flowtext.utils import download_from_url
from flowtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_json,
)
URL = {
    'train': "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SQuAD1/train-v1.1.json",
    'dev': "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SQuAD1/dev-v1.1.json",
}

NUM_LINES = {
    'train': 87599,
    'dev': 10570,
}


DATASET_NAME = "SQuAD1"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def SQuAD1(root, split):
    extracted_files = download_from_url(URL[split], root=root)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_json(extracted_files))
