from pathlib import Path
from polimorfo.datasets.coco import CocoDataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from . import maskutils
import pandas as pd
from sklearn import metrics
import sys

__all__ = [
    "generate_predictions",
    "mean_average_precision_and_recall",
    "mean_average_precision_and_recall_per_class",
]

REPORT_HEADER = [
    "img_path",
    "gt_ann_id",
    "pred_ann_id",
    "true_class_id",
    "pred_class_id",
    "intersection",
    "union",
    "IOU",
    "score",
    "true_area",
    "pred_area",
    "class_value",
]


def __best_match(
    pred_anns: List,
    gt_img_meta: Dict,
    gt_ann_id: int,
    gt_mask: np.ndarray,
    img_path: str,
    gt_class_id: int,
    gt_area: int,
) -> Tuple[int, List]:
    """
    compute the best prediction given the ground truth annotation

    Args:
        pred_anns (List): the list of the annotations for the image
        gt_img_meta (Dict): the metadata of the img
        gt_ann_id (int): the idx of the grount truth annotation
        gt_mask (np.ndarray): the mask for the ground truth
        img_path (str): the path of the image
        gt_class_id (int): the idx of the ground truth class
        gt_area: the area of the gt annotation

    Returns:
        Tuple[int, List]: the id of the best prediction and the values to be saved
    """
    best_pred_ann_id = -1
    best_iou = 0

    # false negative as default
    best_values = [
        img_path,
        gt_ann_id,
        -1,
        gt_class_id,
        -1,
        0,
        0,
        0,
        1,
        gt_area,
        sys.float_info.max,
        "false_negative",
    ]
    for pred_ann in pred_anns:
        pred_mask = maskutils.polygons_to_mask(
            pred_ann["segmentation"], gt_img_meta["height"], gt_img_meta["width"]
        )
        pred_ann_id = pred_ann["id"]
        pred_class_id = pred_ann["category_id"]
        pred_score = pred_ann["score"] if "score" in pred_ann else 1

        intersection = (pred_mask * gt_mask).sum()
        union = np.count_nonzero(pred_mask + gt_mask)
        if intersection == 0 and union == 0:
            iou = 0
        else:
            iou = intersection / union

        if iou > best_iou:
            if pred_class_id == gt_class_id:
                value = "true_positive"
            else:
                value = "false_positive"

            best_values = [
                img_path,
                gt_ann_id,
                pred_ann_id,
                gt_class_id,
                pred_class_id,
                intersection,
                union,
                iou,
                pred_score,
                gt_area,
                pred_ann["area"],
                value,
            ]
            best_pred_ann_id = pred_ann_id
            best_iou = iou
    return best_pred_ann_id, best_values


def generate_predictions_from_ds(
    gt_ds: CocoDataset, pred_ds: CocoDataset, category_idxs: List[int] = None
) -> pd.DataFrame:

    gt_ds.reindex()
    pred_ds.reindex()

    results = []

    for img_idx, gt_img_meta in tqdm(gt_ds.imgs.items()):

        gt_anns = gt_ds.get_annotations(img_idx, category_idxs)
        pred_img_meta = pred_ds.imgs[img_idx]

        if gt_img_meta["file_name"] != pred_img_meta["file_name"]:
            raise Exception("images path compared are different")

        img_path = gt_img_meta["file_name"]

        pred_anns = pred_ds.get_annotations(img_idx, category_idxs)
        # create a set with all the prediction that will be used to find FP
        pred_idx_dict = {ann["id"]: ann for ann in pred_anns}

        if (len(gt_anns) == 0) and (len(pred_anns) == 0):
            results.append(
                [
                    img_path,
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    0,
                    0,
                    1,
                    sys.float_info.max,
                    sys.float_info.max,
                    "true_negative",
                ]
            )

        # iterate of the gr annotations
        for gt_ann in gt_anns:
            gt_mask = maskutils.polygons_to_mask(
                gt_ann["segmentation"], gt_img_meta["height"], gt_img_meta["width"]
            )
            gt_ann_id = gt_ann["id"]
            gt_class_id = gt_ann["category_id"]

            pred_ann_id, row = __best_match(
                pred_anns,
                gt_img_meta,
                gt_ann_id,
                gt_mask,
                img_path,
                gt_class_id,
                gt_ann["area"],
            )
            results.append(row)
            if pred_ann_id in pred_idx_dict:
                del pred_idx_dict[pred_ann_id]
                pred_anns = pred_idx_dict.values()

        # add the false positive
        for pred_ann_id, pred_ann in pred_idx_dict.items():
            # put a false positive with high score in order to not remove it from metrics
            results.append(
                [
                    img_path,
                    -1,
                    pred_ann_id,
                    -1,
                    pred_ann["category_id"],
                    0,
                    0,
                    0,
                    pred_ann["score"],
                    sys.float_info.max,
                    pred_ann["area"],
                    "false_positive",
                ]
            )

    return pd.DataFrame(results, columns=REPORT_HEADER)


def generate_predictions(
    gt_path: str,
    preds_path: str,
    images_path: str = None,
    category_idxs: List[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    create a list that contains the comparison between the predictions
        and the ground truth to be used to compute all the metrics

    Args:
        gt_path (str): the path of the ground truth annotations
        preds_path (str): the path of the prediction annotations
        images_path (str): the path were are saved the images

    Raises:
        Exception: returns an execption if the image idx of the files are not aligned

    Returns:
        pd.DataFrame: [description]
    """
    if images_path is None:
        images_path = Path(gt_path).parent / "images"

    gt_ds = CocoDataset(gt_path, images_path)
    pred_ds = CocoDataset(preds_path, images_path)
    return generate_predictions_from_ds(gt_ds, pred_ds, category_idxs)


def mean_average_precision_and_recall(
    prediction_report: pd.DataFrame,
    range_iou: np.ndarray = np.arange(0.5, 1.0, 0.05),
    min_score: float = 0.0,
) -> Tuple[float, float]:
    """
    compute mean average precision and recall for a given range

    Args:
        prediction_report (pd.DataFrame): a dataframe generated using the method
            :func:`generate_predictions <polimorfo.utils.cocoeval.generate_predictions>`
        range_iou (np.ndarray, optional): a valid range of values. Defaults to np.arange(.5, 1., .05).
        min_score: (float): the min score used to filter annotations. Defaults: 0.0

    Returns:
        Tuple[float, float]: mean average precision and mean average recall
    """

    df = prediction_report
    # filter low score predictions
    df = df[df["score"] >= min_score]
    precisions = []
    recalls = []
    for iou in range_iou:
        true_positives = len(
            df[(df["true_class_id"] == df["pred_class_id"]) & (df["IOU"] >= iou)]
        )
        # all the prediction that do not have a valid gt annotation
        false_positives = len(df[df["gt_ann_id"] == -1])
        # all the gt annotations that do not have a prediction
        false_negatives = len(df[df["pred_ann_id"] == -1])
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / (true_positives + false_negatives))

    if len(precisions) == 0:
        return None, None

    if len(precisions) == 1:
        return precisions[0], recalls[0]

    return np.mean(precisions), np.mean(recalls)


def mean_average_precision_and_recall_per_class(
    prediction_report: pd.DataFrame,
    range_iou: np.ndarray = np.arange(0.5, 1.0, 0.05),
    min_score: float = 0.0,
    idx_class_dict: Dict[int, str] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    generate mean average precision and recall for class idx

    Args:
        prediction_report (pd.DataFrame): a dataframe generated using the method
            :func:`generate_predictions <polimorfo.utils.cocoeval.generate_predictions>`
        range_iou (np.ndarray, optional): a valid range of values. Defaults to np.arange(.5, 1., .05).

    Returns:
        Dict[int, Tuple[float, float]]: a dictionarry of shape class_id -> Tuple[MAP, MAR]
    """

    df = prediction_report
    class_idx_metrics = dict()
    class_idxs = sorted(df["true_class_id"].unique())
    for class_idx in class_idxs:
        if class_idx in [-1, 0]:
            continue
        df_class = df[
            (df["true_class_id"] == class_idx) | (df["pred_class_id"] == class_idx)
        ]

        map_mar = mean_average_precision_and_recall(df_class, range_iou, min_score)
        if idx_class_dict is not None and class_idx in idx_class_dict:
            class_name = idx_class_dict[class_idx]
            class_idx_metrics[class_name] = map_mar
        else:
            class_idx_metrics[class_idx] = map_mar

    return class_idx_metrics


def precision_recall_per_image(
    prediction_report: pd.DataFrame,
    image_name: str,
    range_iou: np.ndarray = np.arange(0.5, 1.0, 0.05),
    min_score: float = 0.0,
) -> Tuple[float, float]:
    df = prediction_report
    df_image = df[df["img_path"] == image_name]
    return mean_average_precision_and_recall(df_image, range_iou, min_score)


def confusion_matrix(
    prediction_report: pd.DataFrame,
    min_iou,
    min_score,
    idx_class_dict: Dict[int, str] = None,
    normalize=False,
) -> Tuple[np.ndarray, metrics.ConfusionMatrixDisplay]:
    df = prediction_report[prediction_report["IOU"] >= min_iou]
    df = df[df["score"] >= min_score]

    class_idxs = sorted(
        list(
            set(
                df["true_class_id"].unique().tolist()
                + df["pred_class_id"].unique().tolist()
            )
        )
    )

    labels = []
    if idx_class_dict is not None:
        for idx in class_idxs:
            if idx == 0:
                labels.append("no_danno")
            else:
                labels.append(idx_class_dict[idx])
    else:
        labels = class_idxs

    cm = metrics.confusion_matrix(df["true_class_id"], df["pred_class_id"])
    if normalize:
        cm = cm / cm.sum()
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
    return cm, cm_display
