import oneflow as flow
import csv
import hashlib
import os
import tarfile
import logging
import sys
import zipfile
import gzip
from ._download_datasets import _DATASET_DOWNLOAD_MANAGER



def download_from_url(url, path=None, root='.data'):
    """
    Args:
        url: the url of the file from URL header. (None)
        path: path where file will be saved
        root: download folder used to store the file in (.data)
    
    Examples:
        >>> url = 'http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/AG_NEWS/train.csv'
        >>> flowtext.utils.download_from_url(url)
        >>> '.data/train.csv'

    """
    if path is None:
        _, filename = os.path.split(url)
        root = os.path.abspath(root)
        path = os.path.join(root, filename)
    else:
        path = os.path.abspath(path)
        root, filename = os.path.split(os.path.abspath(path))

    # skip download if path exists and overwrite is not True
    if os.path.exists(path):
        logging.info('File %s already exists.' % path)
        return path

    # make root dir if does not exist
    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            raise OSError("Can't create the download directory {}.".format(root))

    # download data and move to path
    _DATASET_DOWNLOAD_MANAGER.get_local_path(url, destination=path)

    logging.info('File {} downloaded.'.format(path))

    return path


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples

    Args:
        unicode_csv_data: unicode csv data (see example below)

    Examples:
        >>> from flowtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)

    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line

