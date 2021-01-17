import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image
from pytest import fixture
from tqdm import tqdm

from polimorfo.datasets import CocoDataset

BASE_PATH = Path(__file__).parent.parent / "data"


@fixture
def dataset_file():
    return BASE_PATH / "hair_drier_toaster_bear.json"


@fixture
def coco_test(dataset_file):
    return CocoDataset(dataset_file)


def test_load_coco(dataset_file):
    coco = CocoDataset(dataset_file)

    assert len(coco.cats) == 3


def test_categories_images_count(coco_test):
    images_count = coco_test.count_images_per_category()
    assert len(images_count) == 3
    print(images_count)
    assert images_count == {"toaster": 217, "hair drier": 189, "bear": 960}


def test_categories_annotations_count(coco_test):
    images_count = coco_test.count_annotations_per_category()
    assert len(images_count) == 3
    print(images_count)
    assert images_count == {"toaster": 225, "hair drier": 198, "bear": 1294}


def test_keep_categories(coco_test):
    coco_test.keep_categories([80, 89], True)
    assert len(coco_test.cats) == 2


def test_remap_categories(coco_test):
    coco_test.remap_categories({80: 1})
    assert 1 in coco_test.cats
    assert 80 not in coco_test.cats


def test_dumps(coco_test):
    coco_test.keep_categories([80, 89], True)

    data = coco_test.dumps()
    assert len(data["categories"]) == 2
    assert len(data["images"]) == 406


def test_save_segmentation_masks(coco_test):
    out_path = BASE_PATH / "segments"
    coco_test.save_segmentation_masks(out_path, [23], {23: 24})
    assert len(list(out_path.glob("*.png"))) > 0
    distinct_values = set()
    for png_path in tqdm(list(out_path.glob("*.png"))):
        img = np.array(Image.open(png_path))
        distinct_values = distinct_values.union(set(np.unique(img)))

    assert distinct_values == {0, 24}
    shutil.rmtree(out_path.as_posix())
    (out_path.parent / "cat_idx_dict.json").unlink()


def test_save_images_and_masks(coco_test):
    out_path = BASE_PATH / "saved_images_masks"
    images_path, masks_path = coco_test.save_images_and_masks(out_path, [23], {23: 24})
    # assert len(list(images_path.glob("*.jpg"))) > 0
    assert len(list(masks_path.glob("*.png"))) > 0
    # assert len(list(images_path.glob("*.jpg"))) == len(list(masks_path.glob("*.png")))
    shutil.rmtree(out_path.as_posix())


def test_create_dataset():
    annotations_path = BASE_PATH / "new_coco.json"
    ds = CocoDataset(annotations_path)

    cat_id = ds.add_category("dog", "animal")
    img_id = ds.add_image((BASE_PATH / "test_nodamage.jpg").as_posix(), 100, 100)
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds) == 1
    assert len(ds.anns) == 1

    ds.dump()
    assert annotations_path.exists()

    with open(annotations_path, "r") as f:
        assert len(f.readline()) > 0

    os.remove(annotations_path)


def test_create_dataset_existing():
    ds = CocoDataset(BASE_PATH / "new_coco.json")

    cat_id = ds.add_category("dog", "animal")
    img_id = ds.add_image((BASE_PATH / "test_nodamage.jpg").as_posix(), 100, 100)
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    img_id = ds.add_image((BASE_PATH / "test_nodamage.jpg").as_posix(), 100, 100)
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds.imgs) == 2
    assert len(ds.anns) == 2

    assert ds.cat_id == 2
    assert ds.img_id == 3
    assert ds.ann_id == 3


def test_remove_categories():
    ds = CocoDataset(BASE_PATH / "new_coco.json")
    cat_id = ds.add_category("dog", "animal")
    assert len(ds.cats) == 1
    ds.remove_categories([cat_id])

    assert len(ds.cats) == 0


def test_remove_categories_and_annotations():
    ds = CocoDataset(BASE_PATH / "new_coco.json")
    cat_id = ds.add_category("dog", "animal")
    img_id = ds.add_image((BASE_PATH / "test_nodamage.jpg").as_posix(), 100, 100)
    ds.add_annotation(img_id, cat_id, [1, 2, 3, 4, 5], 10, [0, 0, 256, 256], 0)

    assert len(ds.cats) == 1
    ds.remove_categories([cat_id])

    assert len(ds.cats) == 0


def test_create_split(coco_test):
    train, val, test = coco_test.split(0.8, 0.1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_create_split_notest(coco_test: CocoDataset):
    train, val, test = coco_test.split(0.8, 0.3)
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
    assert coco_test.imgs[1]["file_name"] == "000000001442.jpg"


def test_get_annotations_for_image(coco_test: CocoDataset):
    coco_test.reindex()
    img_idx = 1
    ann_idxs = coco_test.index.imgidx_to_annidxs[img_idx]
    coco_test.remove_annotations(ann_idxs)
    anns = coco_test.get_annotations(img_idx)
    assert isinstance(anns, list)
    assert len(anns) == 0
