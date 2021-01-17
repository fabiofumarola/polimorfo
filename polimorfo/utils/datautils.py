import tarfile
import zipfile
import os
import hashlib
import shutil
import math
import re
from pathlib import Path
from typing import Tuple, List
import requests
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser
import polimorfo

CACHE_DIR = polimorfo.__cache_dir__
CACHE_SUBDIR = "datasets"


def download_file(
    name: str,
    url: str,
    file_hash=None,
    extract: bool = True,
    cache_dir: str = CACHE_DIR,
    cache_subdir: str = CACHE_SUBDIR,
) -> List[Path]:
    """Downloads a file frolsm a URL if it not already saved
    Arguments:
        name {str} -- the name of the file (e.g. )
        url {str} -- the url of the file or the idx of the file in
            case files are download from google drive
    Keyword Arguments:
        file_hash {str} -- the hash of the file downloads (default: {None})
        extract {bool} -- try to extract the file (default: {True})
        cache_dir {str} -- the default folder where the file is saved
            (default: {carambola.utils.datautils.CACHE_DIR})
        cache_subdir {str} -- the subdir where the file is downloaded
            (default: {carambola.utils.datautils.CACHE_SUBDIR})
    """

    if cache_dir is None:
        cache_dir = CACHE_DIR

    if extract:
        assert name.endswith(
            tuple(["zip", "tar", "tar.gz"])
        ), "selected extract=True with a non archive file"

    cache_subdir = Path(cache_dir) / cache_subdir
    cache_subdir = cache_subdir.expanduser()
    cache_subdir.mkdir(parents=True, exist_ok=True)
    # print('saving data in path {}'.format(cache_subdir.absolute()))
    file_path = cache_subdir / name

    download = False
    if file_path.exists():
        if not validate_file(file_path, file_hash):
            download = True
    else:
        download = True

    if download:
        if url.startswith("https://drive.google.com"):
            download_from_gdrive(url, file_path)
        elif url.startswith("http"):
            download_url(url, file_path)
        else:
            download_from_gdrive(url, file_path)

    if extract:
        return extract_archive(file_path, cache_subdir)

    return [file_path]


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    Example:
    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        The file hash
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        Whether the file is valid
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def __save_response_content(response: requests.Response, destination: Path):
    """save the content of the request as file
    Arguments:
        response {requests.Response} -- the response to download
        destination {Path} -- the destination path
    """
    chunk_size = 32768

    num_elems = math.ceil(256483864 / chunk_size)

    with open(destination, "wb") as fh:
        for chunk in tqdm(
            response.iter_content(chunk_size),
            total=num_elems,
            desc="saving file {}".format(destination.name),
        ):
            if chunk:  # filter out keep-alive new chunks
                fh.write(chunk)


def download_from_gdrive(uri: str, dst_path: str) -> Path:
    """download a file/folder from google drive
    Given a url https://drive.google.com/file/d/1EcUzQPNQXQGiHES9gU7oh-886wbBH3VF/view?usp=sharing
    the idx -> 1EcUzQPNQXQGiHES9gU7oh-886wbBH3VF
    Arguments:
        uri {str} -- the id of the file to download or the full google drive url
        dst_path {str} -- the path to save the file
    Returns:
        Path -- the path where the file is saved
    """
    gdrive_resource_re = r"https:\/\/drive\.google.com\/file\/d\/([\w-]+)\/?"
    regex_match = re.match(gdrive_resource_re, uri.strip())
    if regex_match:
        idx = regex_match.groups()[0]
    else:
        idx = uri

    dst_path.parent.mkdir(exist_ok=True, parents=True)

    base_url = "https://docs.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(base_url, params={"id": idx}, stream=True)
        token = __get_confirm_token(response)

        if token:
            params = {"id": idx, "confirm": token}
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
    file_size = int(req.headers["Content-Length"])
    chunk = 1
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)

    with open(dst_path, "wb") as fp:
        for chunk in tqdm(
            req.iter_content(chunk_size=chunk_size),
            total=num_bars,
            unit="KB",
            desc="downloading {}".format(dst_path.name),
            leave=True,
        ):
            fp.write(chunk)
    return file_size


def extract_archive(
    file_path: str, dst_path: str = "", archive_format="auto"
) -> List[Path]:
    """Extract the archive if it match tar, tar.gz, tar.bz or zip format
    Arguments:
        file_path {str} -- the path to the archive
    Keyword Arguments:
        dst_path {str} -- the path to extract the folder (default: {None}
            the directory where the archive is placed)
        archive_format {str} -- The format of the archive (default: {'auto'})
            Options are: 'auto', 'tar', 'zip'
    Returns:
        List[Path] -- the paths where the file is saved
    """
    file_path = Path(file_path)
    if dst_path == "":
        dst_path = file_path.parent
    else:
        dst_path = Path(dst_path)
    dst_path.mkdir(exist_ok=True, parents=True)
    dst_path = dst_path / file_path.name.split(".")[0]

    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == "zip":
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
            list_files = list(dst_path.iterdir())
            # flatten the structure in the case the folder contains just one folder
            if (len(list_files) == 1) and (
                list_files[0].name == file_path.name.split(".")[0]
            ):
                parent = list_files[0].parent
                # copy the content of the file in the original folder
                for file in list_files[0].iterdir():
                    shutil.move(file.as_posix(), parent.as_posix())
                shutil.rmtree(list_files[0].as_posix())

                return [parent]

            return list(dst_path.iterdir())
    # return the input as output
    return [file_path]


if __name__ == "__main__":
    parser = ArgumentParser("compute sha256 of a given file")
    parser.add_argument("--file", type=Path, required=True)

    args = parser.parse_args()
    print(_hash_file(args.file))
