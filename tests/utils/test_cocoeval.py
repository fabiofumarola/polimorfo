from pathlib import Path
from polimorfo.utils import cocoeval

BASE_PATH = Path(__file__).parent.parent / 'data'


def test_generate_predictions():
    ds_path = BASE_PATH / 'hair_drier_toaster_bear.json'
    df = cocoeval.generate_predictions(ds_path, ds_path)
    assert len(df) > 0
    assert df['IOU'].mean() == 1.


def test_mean_average_precision_and_recall():
    ds_path = BASE_PATH / 'hair_drier_toaster_bear.json'
    df = cocoeval.generate_predictions(ds_path, ds_path)
    map, mar = cocoeval.mean_average_precision_and_recall(df)
    assert map == 1.
    assert mar == 1.


def test_mean_average_precision_and_recall_per_class():
    ds_path = BASE_PATH / 'hair_drier_toaster_bear.json'
    df = cocoeval.generate_predictions(ds_path, ds_path)
    class_idx_metrics = cocoeval.mean_average_precision_and_recall_per_class(df)
    class_idxs = df['true_class_id'].unique()
    class_idxs = class_idxs[class_idxs > 0]
    print(class_idx_metrics)
    assert len(class_idx_metrics) == len(class_idxs)