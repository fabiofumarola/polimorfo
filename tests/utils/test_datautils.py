import os
import shutil
from pathlib import Path

import pytest

from polimorfo.utils import datautils

BASE_PATH = Path(__file__).parent.parent / "data"


@pytest.fixture
def cleandir():
    to_cancel = BASE_PATH / "datasets"
    if to_cancel.exists():
        shutil.rmtree(to_cancel)


# def test_download_mnist():
#     dst_path = BASE_PATH / "mnist.zip"
#     file_size = datautils.download_url(
#         url="https://github.com/mlampros/DataSets/raw/master/mnist.zip",
#         dst_path=dst_path,
#     )
#     assert file_size == 15291135
#     assert dst_path.exists()
#     os.remove(dst_path)


# def test_unzip_file():
#     dst_path = BASE_PATH / "mnist.zip"
#     _ = datautils.download_url(
#         url="https://github.com/mlampros/DataSets/raw/master/mnist.zip",
#         dst_path=dst_path,
#     )
#     out_paths = datautils.extract_archive(str(dst_path))
#     assert out_paths[0].name == "mnist.csv"
#     for out in out_paths:
#         shutil.rmtree(out.parent)
#     os.remove(dst_path)


def test_download_gdrivezip():
    dst_path = BASE_PATH / "test.tar.gz"
    out_path = datautils.download_from_gdrive(
        "1Sr1fm8PaaKQuvVpl34t6xyWty61o2psH", dst_path
    )
    assert out_path.name == "test.tar.gz"
    os.remove(out_path)


@pytest.mark.usefixtures("cleandir")
def test_download_archive_gdrive():
    files = datautils.download_file(
        name="test.tar.gz", url="1Sr1fm8PaaKQuvVpl34t6xyWty61o2psH", cache_dir=BASE_PATH
    )

    print(files)
    assert len(files) > 0
    for f in files:
        if f.parent.exists():
            shutil.rmtree(f.parent)


@pytest.mark.usefixtures("cleandir")
def test_download_file_gdrive():
    files = datautils.download_file(
        name="requirements.txt",
        url="1_K6gqtZrvGWsINfkY0J6Evbd3bbWLg_0",
        cache_dir=BASE_PATH,
        extract=False,
    )

    print(files)
    assert len(files) > 0
    for f in files:
        os.remove(f)


@pytest.mark.usefixtures("cleandir")
def test_download_file_extract():
    with pytest.raises(AssertionError):
        datautils.download_file(
            name="requirements.txt",
            url="1_K6gqtZrvGWsINfkY0J6Evbd3bbWLg_0",
            cache_dir=BASE_PATH,
        )


@pytest.mark.usefixtures("cleandir")
def test_download_archive():
    files = datautils.download_file(
        name="mnist.zip",
        url="https://github.com/mlampros/DataSets/raw/master/mnist.zip",
        cache_dir=BASE_PATH,
        extract=True,
    )

    print(files)
    assert len(files) > 0
    shutil.rmtree(files[0].parent)
    shutil.rmtree(files[0].parent.parent)


# @pytest.mark.usefixtures("cleandir")
# def test_download_file_github():
#     files = datautils.download_file(
#         name='densenet121_weights_tf_dim_ordering_tf_kernels.h5',
#         url=
#         'https://github.com/keras-team/keras-applications/releases/download/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
#         cache_dir=BASE_PATH,
#         extract=False)

#     print(files)
#     assert len(files) > 0
#     shutil.rmtree(files[0].parent)
