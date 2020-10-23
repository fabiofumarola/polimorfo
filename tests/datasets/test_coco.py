import pytest
from pytest import fixture
from pathlib import Path
import shutil
from polimorfo.datasets import CocoDataset, ExportFormat

BASE_PATH = Path(__file__).parent.parent / 'data'


@fixture
def dataset_file():
    return BASE_PATH / 'hair_drier_toaster_bear.json'


@fixture
def coco_test(dataset_file):
    return CocoDataset(dataset_file)


def test_load_coco(dataset_file):
    coco = CocoDataset(dataset_file)

    assert len(coco.cats) == 3


def test_categories_images_count(coco_test):
    images_count = coco_test.cats_images_count()
    assert len(images_count) == 3
    assert images_count == {'toaster': 217, 'hair drier': 189, 'bear': 960}


def test_categories_annotations_count(coco_test):
    images_count = coco_test.cats_annotations_count()
    assert len(images_count) == 3
    assert images_count == {'toaster': 225, 'hair drier': 198, 'bear': 1294}


def test_keep_categories(coco_test):
    coco_test.keep_categories([80, 89], True)
    assert len(coco_test.cats) == 2


def test_dumps(coco_test):
    coco_test.keep_categories([80, 89], True)

    data = coco_test.dumps()
    assert len(data['categories']) == 2
    assert len(data['images']) == 406


def test_dump_segmentation(coco_test):
    out_path = BASE_PATH / 'segments'
    coco_test.dump(out_path, ExportFormat.segmentation)
    assert len(list(out_path.glob('*.png'))) > 0
    shutil.rmtree(out_path.as_posix())


def test_create_dataset():
    ds = CocoDataset()

    cat_id = ds.add_category('dog', 'animal')
    img_id = ds.add_image((BASE_PATH / 'test_nodamage.jpg').as_posix())
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds) == 1
    assert len(ds.anns) == 1


def test_create_dataset_existing():
    ds = CocoDataset()

    cat_id = ds.add_category('dog', 'animal')
    img_id = ds.add_image((BASE_PATH / 'test_nodamage.jpg').as_posix())
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    img_id = ds.add_image((BASE_PATH / 'test_nodamage.jpg').as_posix())
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds.imgs) == 2
    assert len(ds.anns) == 2

    assert ds.cat_id == 2
    assert ds.img_id == 2
    assert ds.ann_id == 2


def test_remove_categories():
    ds = CocoDataset()
    cat_id = ds.add_category('dog', 'animal')
    assert len(ds.cats) == 1
    ds.remove_categories([cat_id])

    assert len(ds.cats) == 0


def test_remove_categories_and_annotations():
    ds = CocoDataset()
    cat_id = ds.add_category('dog', 'animal')
    img_id = ds.add_image((BASE_PATH / 'test_nodamage.jpg').as_posix())
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds.cats) == 1
    ds.remove_categories([cat_id])

    assert len(ds.cats) == 0


def test_create_split(coco_test):
    train, val, test = coco_test.split(.8, .1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_create_split_notest(coco_test: CocoDataset):
    train, val, test = coco_test.split(.8, .3)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) == 0


def test_copy(coco_test: CocoDataset):
    copy_coco = coco_test.copy()
    assert id(coco_test.imgs) != id(copy_coco.imgs)
    assert id(coco_test.cats) != id(copy_coco.cats)
    assert id(coco_test.anns) != id(copy_coco.anns)


def test_update_images_path(coco_test: CocoDataset):
    coco_test.update_images_path(lambda x: Path(x).name)
    coco_test.reindex()
    assert coco_test.imgs[1]['file_name'] == '000000410627.jpg'