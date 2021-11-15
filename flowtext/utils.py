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


def download_from_url(url, path=None, root=".data"):
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
        logging.info("File %s already exists." % path)
        return path

    # make root dir if does not exist
    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            raise OSError("Can't create the download directory {}.".format(root))

    # download data and move to path
    _DATASET_DOWNLOAD_MANAGER.get_local_path(url, destination=path)

    logging.info("File {} downloaded.".format(path))

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


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/Multi30k/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> flowtext.utils.download_from_url(url, from_path)
        >>> flowtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> flowtext.utils.download_from_url(url, from_path)
        >>> flowtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {}.".format(from_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {}.".format(from_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in zfile.namelist():
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {}.".format(from_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )
