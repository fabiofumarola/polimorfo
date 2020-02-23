from polimorfo.datasets import Coco
from pytest import fixture
from pathlib import Path
import shutil
import pytest

BASE_PATH = Path(__file__).parent.parent / 'data'


@fixture
def dataset_file():
    return BASE_PATH / 'hair_drier_toaster_bear.json'


@fixture
def coco_test(dataset_file):
    return Coco(dataset_file)


def test_load_coco(dataset_file):
    coco = Coco(dataset_file)

    assert len(coco.get_categories()) == 3


def test_get_categories(coco_test):
    assert isinstance(coco_test.get_categories(True), dict)
    assert isinstance(coco_test.get_categories(False), list)


def test_categories_images_count(coco_test):
    images_count = coco_test.categories_images_count()
    assert len(images_count) == 3
    assert images_count == [('bear', 960), ('toaster', 217),
                            ('hair drier', 189)]


def test_categories_annotations_count(coco_test):
    images_count = coco_test.categories_annotations_count()
    assert len(images_count) == 3
    assert images_count == [('bear', 1294), ('toaster', 225),
                            ('hair drier', 198)]


def test_keep_categories_id(coco_test):
    coco_test.keep_categories_id([80, 89])
    assert coco_test.to_keep_id_categories == {80, 89}


def test_keep_categories_name(coco_test):
    coco_test.keep_categories_name(['bear', 'toaster'])
    assert coco_test.to_keep_id_categories == {23, 80}


def test_dumps(coco_test):
    coco_test.keep_categories_name(['bear', 'toaster'])

    data = coco_test.dumps()
    assert len(data['categories']) == 2
    assert len(data['images']) == 1177


def test_download_coco2017():
    train_data, val_data = Coco.download_data(task='object_detection',
                                              version='2017',
                                              path='tests/data')
    assert len(train_data) == 118287
    assert len(val_data) == 5000
    print('train data len={}'.format(len(train_data)))
    print('val data len={}'.format(len(val_data)))
    shutil.rmtree('tests/data/datasets')


def test_download_coco2014():
    train_data, val_data = Coco.download_data(task='object_detection',
                                              version='2014',
                                              path='tests/data')

    assert len(train_data) == 118287
    assert len(val_data) == 5000
    print('train data len={}'.format(len(train_data)))
    print('val data len={}'.format(len(val_data)))
    shutil.rmtree('tests/data/datasets')


def test_download_coco2017_captioning():
    with pytest.raises(NotImplementedError):
        Coco.download_data(task='captioning',
                           version='2017',
                           path='tests/data')


def test_download_coco2017_keypoints():
    with pytest.raises(NotImplementedError):
        Coco.download_data(task='keypoints', version='2017', path='tests/data')


def test_download_coco2017_wrong_name():
    with pytest.raises(ValueError):
        Coco.download_data(task='panoptic', version='2017', path='tests/data')


def test_download_coco2017_wrong_year():
    with pytest.raises(ValueError):
        Coco.download_data(task='object_detection',
                           version='2013',
                           path='tests/data')
