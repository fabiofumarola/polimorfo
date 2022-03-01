import cv2
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.pyplot import contour

__all__ = [
    "mask_to_polygon",
    "polygons_to_mask",
    "area",
    "bbox",
    "coco_poygons_to_mask",
]


def mask_to_polygon(
    mask, min_score: float = 0.5, approx: float = 0.0, relative: bool = True
):
    """generate polygons from masks

    Args:
        mask (np.ndarray): a binary mask
        min_score (float, optional): [description]. Defaults to 0.5.
        approx (float, optional): it approximate the polygons to reduce the number of points. Defaults to 0.0
        relative (bool, optional): it the value of the approximation is computed on the relative amount of point or with respect to all the points

    Returns:
        [type]: [description]
    """
    mask = (mask > min_score).astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(-1, -1)
    )
    polygons = []
    for cnt in contours:
        if approx > 0:
            if relative:
                epsilon = approx * cv2.arcLength(cnt, True)
            else:
                epsilon = approx
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            approx_poly = cnt
        # we need to draw a least a box
        if len(approx_poly) >= 4:
            approx_flattened = approx_poly.flatten().tolist()
            polygons.append(approx_flattened)
    return polygons


def polygons_to_mask(polygons, height, width):
    """convert polygons to mask. Filter all the polygons with less than 4 points

    Args:
        polygons ([type]): [description]
        height ([type]): [description]
        width ([type]): [description]

    Returns:
        [type]: a mask of format num_classes, heigth, width
    """
    polygons = [polygon for polygon in polygons if len(polygon) >= 8]

    if len(polygons) == 0:
        return np.zeros((height, width), np.uint8)

    rle = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rle)
    return mask_util.decode(rle)[:, :]


def area(mask, min_score=0.5):
    mask = (mask > min_score).astype(np.uint8)
    return int(mask.sum())


def bbox(
    polygons,
    height,
    width,
):
    p = mask_util.frPyObjects(polygons, height, width)
    p = mask_util.merge(p)
    bbox_xywh = mask_util.toBbox(p)
    return bbox_xywh


def bbox_from_mask(mask):
    """return the bounding box from the given mask

    Args:
        mask ([type]): [description]

    Returns:
        List: a list of format [x_min, y_min, w, h]
    """
    pairs = np.argwhere(mask == True)
    if len(pairs) == 0:
        return None, None, None, None
    min_row = min(pairs[:, 0])
    max_row = max(pairs[:, 0])
    min_col = min(pairs[:, 1])
    max_col = max(pairs[:, 1])
    w = max_col - min_col
    h = max_row - min_row
    return [float(min_col), float(min_row), float(w), float(h)]


def coco_poygons_to_mask(segmentations, height, width) -> np.ndarray:
    masks = []
    for polygons in segmentations:
        mask = polygons_to_mask(polygons, height, width)
        # mask = np.any(mask, axis=2)
        masks.append(mask)

    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((height, width), dtype=np.uint8)
    return masks
