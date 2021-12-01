from ..utils import download_from_url, extract_archive
from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os

URL = "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/YahooAnswers/yahoo_answers_csv.tar.gz"

NUM_LINES = {
    "train": 1400000,
    "test": 60000,
}

_PATH = "yahoo_answers_csv.tar.gz"

DATASET_NAME = "YahooAnswers"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=10)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def YahooAnswers(root, split):
    dataset_tar = download_from_url(URL, root=root, path=os.path.join(root, _PATH))
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split + ".csv", extracted_files)
    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], _create_data_from_csv(path)
    )
