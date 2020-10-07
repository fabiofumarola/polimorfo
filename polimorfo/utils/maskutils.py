# import cv2
import pycocotools.mask as mask_util
import numpy as np
from skimage import measure


def mask_to_polygons(mask):
    if len(mask.shape) == 3:
        mask = np.squeeze(mask)
    mask = (mask > 0.5).astype(np.uint8)
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(
        mask)    # some versions of cv2 does not support incontiguous arr
    res = measure.find_contours(mask, 0.5)
    hierarchy = res[-1]
    if hierarchy is None:    # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    return res, has_holes


def polygons_to_mask(polygons, height, width):
    rle = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rle)
    return mask_util.decode(rle)[:, :]


def area(mask):
    return mask.sum()


def bbox(polygons, height, width):
    p = mask_util.frPyObjects(polygons, height, width)
    p = mask_util.merge(p)
    bbox = mask_util.toBbox(p)
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox