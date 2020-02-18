import requests
from tqdm.autonotebook import tqdm
from pathlib import Path
from typing import Tuple, List
import tarfile
import zipfile
import os
import shutil
import math


def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def __save_response_content(response, destination):
    chunk_size = 32768

    num_elems = math.ceil(256483864 / chunk_size)

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=num_elems):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_from_gdrive(idx: str, dst_path: str) -> Path:
    """download a file/folder from google drive

    Given a url https://drive.google.com/file/d/1EcUzQPNQXQGiHES9gU7oh-886wbBH3VF/view?usp=sharing
    the idx -> 1EcUzQPNQXQGiHES9gU7oh-886wbBH3VF
    
    Arguments:
        idx {str} -- the id of the file to download
        dst_path {str} -- the path to save the file
    
    Returns:
        Path -- the path where the file is saved
    """

    dst_path.parent.mkdir(exist_ok=True, parents=True)

    base_url = "https://docs.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(base_url, params={'id': idx}, stream=True)
        token = __get_confirm_token(response)

        if token:
            params = {'id': idx, 'confirm': token}
            response = session.get(base_url, params=params, stream=True)

        __save_response_content(response, dst_path)

    dst_path = Path(dst_path)
    return dst_path


def download_url(url: str, dst_path: str) -> Tuple[Path, int]:
    """download a url to the destination folder
    
    Arguments:
        url {str} -- [description]
        dst_path {str} -- [description]
    
    Returns:
        Tuple[Path, int] -- [description]
    """
    dst_path = Path(dst_path)

    req = requests.get(url, stream=True)
    file_size = int(req.headers['Content-Length'])
    chunk = 1
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)

    with open(dst_path, 'wb') as fp:
        for chunk in tqdm(req.iter_content(chunk_size=chunk_size),
                          total=num_bars,
                          unit='KB',
                          desc=dst_path.name,
                          leave=True):
            fp.write(chunk)
    return file_size


def extract_archive(file_path: str,
                    dst_path: str = '',
                    archive_format='auto') -> List[Path]:
    """Extract the archive if it match tar, tar.gz, tar.bz or zip format
    
    Arguments:
        file_path {str} -- the path to the archive
    
    Keyword Arguments:
        dst_path {str} -- the path to extract the folder (default: {None} the directory where the archive is placed)
        archive_format {str} -- The format of the archive (default: {'auto'})
            Options are: 'auto', 'tar', 'zip'
    
    Returns:
        List[Path] -- the paths where the file is saved
    """
    file_path = Path(file_path)
    if dst_path == '':
        dst_path = file_path.parent
    else:
        dst_path = Path(dst_path)
    dst_path.mkdir(exist_ok=True, parents=True)
    dst_path = dst_path / file_path.name.split('.')[0]

    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as fh:
                try:
                    fh.extractall(dst_path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if dst_path.exists():
                        if dst_path.is_file():
                            os.remove(dst_path)
                        else:
                            shutil.rmtree(dst_path)
                    raise
            return list(dst_path.iterdir())
    # return the input as output
    return [file_path]


def download_file(name: str,
                  url: str,
                  md5_hash=None,
                  extract: bool = True,
                  cache_dir: str = '~/.polimorfo',
                  cache_subdir: str = 'datasets') -> List[Path]:
    """Downloads a file frolsm a URL if it not already saved

    Arguments:
        name {str} -- the name of the file
        url {str} -- the url of the file or the idx of the file in case files are download from google drive

    Keyword Arguments:
        md5_hash {str} -- the hash of the file downloads (default: {None})
        extract {bool} -- try to extract the file (default: {True})
        cache_dir {str} -- the default folder where the file is saved
            (default: {'~/.polimorfo'})
        cache_subdir {str} -- the subdir where the file is downloaded (default: {'datasets'})
    """

    if extract:
        assert name.endswith(tuple(
            ['zip', 'tar',
             'tar.gz'])), 'selected extract=True with a non archive file'

    cache_subdir = Path(cache_dir) / cache_subdir
    cache_subdir.mkdir(parents=True, exist_ok=True)
    file_path = cache_subdir / name

    if url.startswith('http'):
        download_url(url, file_path)
    else:
        download_from_gdrive(url, file_path)

    if extract:
        files = extract_archive(file_path, cache_subdir)
        return files

    return [file_path]
