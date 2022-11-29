import hashlib
import os
import tarfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import tempfile
import shutil
from tqdm import tqdm

import oneflow as flow


def load_state_dict_from_url(url: str, saved_path: str):
    if saved_path == None:
        saved_path = "./pretrained_flow"
    url_parse = urlparse(url)
    file_name = url_parse.path.split("/")[-1].split(".")[0]
    package_name = url_parse.path.split("/")[-1]
    file_path = os.path.join(saved_path, file_name)
    package_path = os.path.join(saved_path, package_name)
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    if not os.path.exists(file_path):
        if not os.path.exists(package_path):
            download_url_to_file(url, package_path)
            print(
                "The pretrained-model file saved in '{}'".format(
                    os.path.abspath(saved_path)
                )
            )
            with tarfile.open(package_path) as f:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, saved_path)
    cpt = get_cpt(file_path)
    config_file = os.path.join(file_path, "config.json")
    vocab_file = os.path.join(file_path, "vocab.txt")
    assert os.path.isdir(cpt), "Checkpoint file error!"
    return flow.load(cpt), config_file, vocab_file


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    file_size = None
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_state_dict_from_file(checkpoint_path):
    cpt = get_cpt(checkpoint_path)
    print(checkpoint_path)
    config_file = os.path.join(checkpoint_path, "config.json")
    vocab_file = os.path.join(checkpoint_path, "vocab.txt")
    assert flow.load(cpt), "The checkpoint_path error."
    return flow.load(cpt), config_file, vocab_file


def get_cpt(file_path):
    files = os.listdir(file_path)
    for f in files:
        cpt = os.path.join(file_path, f)
        if os.path.isdir(cpt):
            return cpt
