from ..utils import download_from_url, extract_archive
from ..data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

import io
from pathlib import Path

URL = "http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/IMDB/aclImdb_v1.tar.gz"

NUM_LINES = {
    "train": 25000,
    "test": 25000,
}

_PATH = "aclImdb_v1.tar.gz"

DATASET_NAME = "IMDB"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def IMDB(root, split):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            *_, split, label, file = Path(fname).parts

            if key == split and (label in ["pos", "neg"]):
                with io.open(fname, encoding="utf8") as f:
                    yield label, f.read()

    dataset_tar = download_from_url(URL, root=root)
    extracted_files = extract_archive(dataset_tar)
    iterator = generate_imdb_data(split, extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], iterator)
