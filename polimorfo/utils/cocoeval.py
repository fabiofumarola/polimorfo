from polimorfo.datasets.coco import CocoDataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from . import maskutils
import pandas as pd

__all__ = [
    'generate_predictions', 'mean_average_precision_and_recall',
    'mean_average_precision_and_recall_per_class'
]


def __best_match(pred_anns: List, gt_img_meta: Dict, gt_ann_id: int,
                 gt_mask: np.ndarray, img_path: str,
                 gt_class_id: int) -> Tuple[int, List]:
    """
    compute the best prediction given the ground truth annotation

    Args:
        pred_anns (List): the list of the annotations for the image
        gt_img_meta (Dict): the metadata of the img
        gt_ann_id (int): the idx of the grount truth annotation
        gt_mask (np.ndarray): the mask for the ground truth
        img_path (str): the path of the image
        gt_class_id (int): the idx of the ground truth class

    Returns:
        Tuple[int, List]: the id of the best prediction and the values to be saved
    """
    best_pred_ann_id = -1
    best_iou = 0
    #
    best_values = [img_path, gt_ann_id, -1, gt_class_id, 0, 0, 0, 0, 0]
    for pred_ann in pred_anns:
        pred_mask = maskutils.polygons_to_mask(pred_ann['segmentation'],
                                               gt_img_meta['height'],
                                               gt_img_meta['width'])
        pred_ann_id = pred_ann['id']
        pred_class_id = pred_ann['category_id']
        pred_score = pred_ann['score'] if 'score' in pred_ann else 1

        intersection = (pred_mask * gt_mask).sum()
        union = np.count_nonzero(pred_mask + gt_mask)
        iou = intersection / union

        if iou > best_iou:
            best_values = [
                img_path, gt_ann_id, pred_ann_id, gt_class_id, pred_class_id,
                intersection, union, iou, pred_score
            ]
            best_pred_ann_id = pred_ann_id
            best_iou = iou
    return best_pred_ann_id, best_values


def generate_predictions(gt_path: str, preds_path: str,
                         **kwargs) -> pd.DataFrame:
    """
    create a list that contains the comparison between the predictions
        and the ground truth to be used to compute all the metrics

    Args:
        gt_path (str): the path of the ground truth annotations
        preds_path (str): the path of the prediction annotations

    Raises:
        Exception: returns an execption if the image idx of the files are not aligned

    Returns:
        pd.DataFrame: [description]
    """
    gt_ds = CocoDataset(gt_path)
    gt_ds.reindex()
    pred_ds = CocoDataset(preds_path)
    pred_ds.reindex()

    header = [
        'img_path', 'gt_ann_id', 'pred_ann_id', 'true_class_id',
        'pred_class_id', 'intersection', 'union', 'IOU', 'score'
    ]

    results = []

    for img_idx, gt_img_meta in tqdm(gt_ds.imgs.items()):
        gt_anns = gt_ds.get_annotations(img_idx)
        pred_img_meta = pred_ds.imgs[img_idx]

        if gt_img_meta['file_name'] != pred_img_meta['file_name']:
            raise Exception("images path compared are different")

        img_path = gt_img_meta['file_name']

        pred_anns = pred_ds.get_annotations(img_idx)
        # create a set with all the prediction that will be used to find FP
        pred_idx_dict = {ann['id']: ann for ann in pred_anns}
        # iterate of the gr annotations
        for gt_ann in gt_anns:
            gt_mask = maskutils.polygons_to_mask(gt_ann['segmentation'],
                                                 gt_img_meta['height'],
                                                 gt_img_meta['width'])
            gt_ann_id = gt_ann['id']
            gt_class_id = gt_ann['category_id']

            pred_ann_id, row = __best_match(pred_anns, gt_img_meta, gt_ann_id,
                                            gt_mask, img_path, gt_class_id)
            results.append(row)
            if pred_ann_id in pred_idx_dict:
                del pred_idx_dict[pred_ann_id]
                pred_anns = pred_idx_dict.values()

        # add the false positive
        for pred_ann_id, pred_ann in pred_idx_dict.items():
            results.append([
                img_path, -1, pred_ann_id, 0, pred_ann['category_id'], 0, 0, 0,
                0
            ])

    return pd.DataFrame(results, columns=header)


def mean_average_precision_and_recall(
    prediction_report: pd.DataFrame,
    range_iou: np.ndarray = np.arange(.5, 1., .05)
) -> Tuple[float, float]:
    """
    compute mean average precision and recall for a given range

    Args:
        prediction_report (pd.DataFrame): a dataframe generated using the method
            :func:`generate_predictions <polimorfo.utils.cocoeval.generate_predictions>`
        range_iou (np.ndarray, optional): a valid range of values. Defaults to np.arange(.5, 1., .05).

    Returns:
        Tuple[float, float]: mean average precision and mean average recall
    """

    df = prediction_report
    precisions = []
    recalls = []
    for iou in range_iou:
        true_positives = len(df[(df['true_class_id'] == df['pred_class_id']) &
                                (df['IOU'] >= iou)])
        # all the prediction that do not have a valid gt annotation
        false_positives = len(df[df['gt_ann_id'] == -1])
        # all the gt annotations that do not have a prediction
        false_negatives = len(df[df['pred_ann_id'] == -1])
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / (true_positives + false_negatives))

    if len(precisions) == 0:
        return None, None

    if len(precisions) == 1:
        return precisions[0], recalls[0]

    return np.mean(precisions), np.mean(recalls)


def mean_average_precision_and_recall_per_class(
    prediction_report: pd.DataFrame,
    range_iou: np.ndarray = np.arange(.5, 1., .05)
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
    class_idxs = sorted(df['true_class_id'].unique())
    for class_idx in class_idxs:
        if class_idx in [-1, 0]:
            continue
        df_class = df[(df['true_class_id'] == class_idx) |
                      (df['pred_class_id'] == class_idx)]
        class_idx_metrics[class_idx] = mean_average_precision_and_recall(
            df_class, range_iou)

    return class_idx_metrics
