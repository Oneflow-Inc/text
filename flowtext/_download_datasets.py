import requests
from tqdm import tqdm


def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get('Content-length', 0))
    with tqdm(total=total_size, unit='B', unit_scale=1) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                t.update(len(chunk))
                yield chunk


def _get_response_from_oneflow(url):
    session = requests.Session()
    response = session.get(url, stream=True)
    filename = url.split('/')[-1]
    print(filename)
    return response, filename


class DownloadManager:
    def get_local_path(self, url, destination):
        response, filename = _get_response_from_oneflow(url)
        with open(destination, 'wb') as f:
            for chunk in _stream_response(response):
                f.write(chunk)


_DATASET_DOWNLOAD_MANAGER = DownloadManager()
