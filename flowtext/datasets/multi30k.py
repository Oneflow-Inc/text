import os
from ..data.datasets_utils import (
    _download_extract,
    _RawTextIterableDataset,
    _wrap_split_argument,
    _create_dataset_directory,
    _read_text_iterator,
)

URL = {
    "train": r"http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/Multi30k/training.tar.gz",
    "valid": r"http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/Multi30k/validation.tar.gz",
    "test": r"http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/Multi30k/mmt16_task1_test.tar.gz",
}


_EXTRACTED_FILES_INFO = {
    "train": {"file_prefix": "train",},
    "valid": {"file_prefix": "val",},
    "test": {"file_prefix": "test",},
}

NUM_LINES = {
    "train": 29000,
    "valid": 1014,
    "test": 1000,
}

DATASET_NAME = "Multi30k"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def Multi30k(root, split, language_pair=("de", "en")):
    """Multi30k dataset

    Reference: http://www.statmt.org/wmt16/multimodal-task.html#task1

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')
    """

    assert (
        len(language_pair) == 2
    ), "language_pair must contain only 2 elements: src and tgt language respectively"
    assert tuple(sorted(language_pair)) == (
        "de",
        "en",
    ), "language_pair must be either ('de','en') or ('en', 'de')"

    downloaded_file = os.path.basename(URL[split])

    src_path = _download_extract(
        root,
        URL[split],
        os.path.join(root, downloaded_file),
        os.path.join(
            root, _EXTRACTED_FILES_INFO[split]["file_prefix"] + "." + language_pair[0]
        ),
    )
    trg_path = _download_extract(
        root,
        URL[split],
        os.path.join(root, downloaded_file),
        os.path.join(
            root, _EXTRACTED_FILES_INFO[split]["file_prefix"] + "." + language_pair[1]
        ),
    )

    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)

    return _RawTextIterableDataset(
        DATASET_NAME, NUM_LINES[split], zip(src_data_iter, trg_data_iter)
    )
