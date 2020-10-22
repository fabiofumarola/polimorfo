from polimorfo.utils import maskutils
from matplotlib.pyplot import sci
from .coco import CocoDataset
from typing import Any, List
import numpy as np
import scipy
from ..utils import maskutils


class SemanticCocoDataset(CocoDataset):

    def add_semantic_annotation(self,
                                img_id: int,
                                masks: np.ndarray,
                                probs: np.ndarray,
                                cats_idxs: List = None,
                                start_index=1) -> int:

        if len(masks.shape) != 2:
            raise ValueError('masks.shape should equal to 2')

        if len(probs.shape) != 3:
            raise ValueError('masks.shape should equal to 3')

        # for each class
        for i, class_idx in enumerate(
                np.unique(masks)[start_index:], start_index):
            #get the mask transposed to match shape (Width, Heigth)
            class_mask = (masks == class_idx).astype(np.uint8)
            class_probs = probs[i]
            if cats_idxs:
                cat_id = cats_idxs[i]
            else:
                cat_id = class_idx

            groups, n_groups = scipy.ndimage.label(class_mask)
            annotation_ids = []
            # get the groups starting from label 1
            for group_idx in range(1, n_groups + 1):
                group_mask = (groups == group_idx).astype(np.uint8)
                polygons = maskutils.mask_to_polygon(group_mask)
                bbox = maskutils.bbox(polygons, *masks.shape)
                area = maskutils.area(group_mask)
                score = np.mean(class_mask * class_probs)
                annotation_ids.append(
                    self.add_annotation(img_id, cat_id, polygons, area, bbox, 0,
                                        score))
            return annotation_ids

    # def add_annotation_from_scores(self,
    #                                img_id: int,
    #                                mask_logits: np.ndarray,
    #                                cats_mapping: List = None,
    #                                start_index=0) -> int:
    #     pass