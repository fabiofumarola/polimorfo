from pathlib import Path
from polimorfo.datasets.coco import CocoDataset
from polimorfo.utils import cocoeval
import numpy as np

BASE_PATH = Path(__file__).parent.parent / "data"


def test_generate_predictions():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"
    df = cocoeval.generate_predictions(ds_path, ds_path)
    assert len(df) > 0
    assert df["IOU"].mean() == 1.0


def test_generate_predictions_one_class():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"
    df = cocoeval.generate_predictions(ds_path, ds_path, category_idxs=[1])
    assert len(df) > 0
    assert df["IOU"].mean() > 0.0


def test_mean_average_precision_and_recall():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"
    df = cocoeval.generate_predictions(ds_path, ds_path)
    map, mar = cocoeval.mean_average_precision_and_recall(df)
    assert map == 1.0
    assert mar == 1.0


def test_mean_average_precision_and_recall_per_class():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"
    df = cocoeval.generate_predictions(ds_path, ds_path)
    class_idx_metrics = cocoeval.mean_average_precision_and_recall_per_class(df)
    class_idxs = df["true_class_id"].unique()
    class_idxs = class_idxs[class_idxs > 0]
    print(class_idx_metrics)
    assert len(class_idx_metrics) == len(class_idxs)


def test_mean_average_precision_and_recall_per_class_with_name():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"

    ds = CocoDataset(ds_path)
    idx_class_name = {idx: cat_meta["name"] for idx, cat_meta in ds.cats.items()}

    df = cocoeval.generate_predictions(ds_path, ds_path)
    class_idx_metrics = cocoeval.mean_average_precision_and_recall_per_class(
        df, idx_class_dict=idx_class_name
    )
    class_idxs = df["true_class_id"].unique()
    class_idxs = class_idxs[class_idxs > 0]
    print(class_idx_metrics)
    assert len(class_idx_metrics) == len(class_idxs)


def test_mean_average_precision_and_recall_per_class_with_name_largest_range():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"

    ds = CocoDataset(ds_path)
    idx_class_name = {idx: cat_meta["name"] for idx, cat_meta in ds.cats.items()}

    df = cocoeval.generate_predictions(ds_path, ds_path)
    class_idx_metrics = cocoeval.mean_average_precision_and_recall_per_class(
        df, idx_class_dict=idx_class_name, range_iou=np.arange(0, 1, 0.5)
    )
    class_idxs = df["true_class_id"].unique()
    class_idxs = class_idxs[class_idxs > 0]
    print(class_idx_metrics)
    assert len(class_idx_metrics) == len(class_idxs)


def test_mean_average_precision_and_recall_per_class_with_name_min_score():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"

    ds = CocoDataset(ds_path)
    idx_class_name = {idx: cat_meta["name"] for idx, cat_meta in ds.cats.items()}

    df = cocoeval.generate_predictions(ds_path, ds_path)
    class_idx_metrics = cocoeval.mean_average_precision_and_recall_per_class(
        df, idx_class_dict=idx_class_name, min_score=0.5
    )
    class_idxs = df["true_class_id"].unique()
    class_idxs = class_idxs[class_idxs > 0]
    print(class_idx_metrics)
    assert len(class_idx_metrics) == len(class_idxs)


def test_precision_recall_per_image():
    ds_path = BASE_PATH / "hair_drier_toaster_bear.json"
    df = cocoeval.generate_predictions(ds_path, ds_path)

    map, mar = cocoeval.precision_recall_per_image(df, df["img_path"].loc[0])
    assert map > 0
    assert mar > 0
