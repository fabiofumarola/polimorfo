from polimorfo.datasets import Coco
from pytest import fixture
from pathlib import Path

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
