from pathlib import Path

import numpy as np
from skmultilearn import model_selection
from tqdm.auto import tqdm

from .coco import CocoDataset


class Splitter:
    @staticmethod
    def split(
        ds: CocoDataset, train_perc: float, val_perc: float, test_perc: float = None
    ):

        if test_perc is None:
            test_perc = 1 - (train_perc + val_perc)
        if not int(train_perc + val_perc + test_perc) == 1:
            raise ValueError(
                "the sum of train val and test percentage is not equal to 1"
            )

        X = []
        y = []
        for img_idx, anns_idx in tqdm(ds.index.imgidx_to_annidxs.items()):
            X.append(img_idx)
            one_hot_vector = [0] * (len(ds.cats) + 1)
            for ann_idx in anns_idx:
                one_hot_vector[ds.anns[ann_idx]["category_id"]] = 1.0
            y.append(one_hot_vector)

        X = np.array(X).reshape((-1, 1))
        y = np.array(y)

        (
            X_train,
            y_train,
            X_val_test,
            y_val_test,
        ) = model_selection.iterative_train_test_split(X, y, val_perc + test_perc)

        if test_perc > 0:
            val_perc_rescaled = val_perc / (val_perc + test_perc)
            X_val, y_val, X_test, y_test = model_selection.iterative_train_test_split(
                X_val_test, y_val_test, val_perc_rescaled
            )
        else:
            X_val, y_val = X_val_test, y_val_test
            X_test, y_test = None, None

        X_train_set = set(X_train.flatten().tolist())
        X_val_set = set(X_val.flatten().tolist())
        X_test_set = set(X_test.flatten().tolist()) if X_test is not None else set()

        split_content = []
        for img_idx, img_meta in ds.imgs.items():
            img_name = Path(img_meta["file_name"]).name
            if img_idx in X_train_set:
                split_content.append(f"{img_name},train\n")
            elif img_idx in X_val_set:
                split_content.append(f"{img_name},val\n")
            else:
                split_content.append(f"{img_name},test\n")

        train_ds = ds.copy()
        train_ds.remove_images(X_val_set | X_test_set)

        val_ds = ds.copy()
        val_ds.remove_images(X_train_set | X_test_set)

        test_ds = ds.copy()
        test_ds.remove_images(X_train_set | X_val_set)

        return split_content, train_ds, val_ds, test_ds
