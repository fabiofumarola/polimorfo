from polimorfo.utils.mergeutils import merge_datasets, get_img_meta_by_name
from polimorfo.datasets import CocoDataset
from pathlib import Path
import numpy as np

BASE_PATH = Path(__file__).parent.parent / "data"


def test_merge_datasets():

    datasets = [
        CocoDataset(coco_path=BASE_PATH / dataset_name)
        for dataset_name in ["dataset1.json", "dataset2.json"]
    ]

    merged_ds = merge_datasets(datasets, BASE_PATH / "fake_merge.json")

    assert len(merged_ds.anns) == 20

    for k in ["scratch", "dent"]:
        assert merged_ds.count_images_per_category()[k] == np.sum(
            [ds_item.count_images_per_category()[k] for ds_item in datasets]
        )
        assert merged_ds.count_annotations_per_category()[k] == np.sum(
            [ds_item.count_annotations_per_category()[k] for ds_item in datasets]
        )

    #  check image meta/ anns consistency
    for ds_item in datasets:
        for img_idx, img_meta in ds_item.imgs.items():

            anns = ds_item.get_annotations(img_idx)

            merged_img_meta = get_img_meta_by_name(merged_ds, img_meta["file_name"])
            merged_img_idx = merged_img_meta["id"]
            merged_anns = merged_ds.get_annotations(merged_img_idx)
            #  number of ann per image
            assert len(merged_anns) == len(anns)
            # number of cats per image
            merged_cats = {a["category_id"] for a in merged_anns}
            cats = {a["category_id"] for a in anns}
            assert len(merged_cats) == len(cats)
            # number of annotated points per image
            merged_segm_points = {len(a["segmentation"]) for a in merged_anns}
            segm_points = {len(a["segmentation"]) for a in anns}
            assert len(merged_segm_points) == len(segm_points)
            # annotation areas per image
            merged_areas = {a["area"] for a in merged_anns}
            areas = {a["area"] for a in anns}
            assert merged_areas == areas
