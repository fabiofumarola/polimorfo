from typing import List

import numpy as np
import scipy
from deprecated import deprecated
from scipy import special

from ..utils import maskutils
from .coco import CocoDataset

__all__ = ["SemanticCoco", "SemanticCocoDataset"]


class SemanticCoco(CocoDataset):
    """
    An extension of the coco dataset to handle the output of a semantic segmentation model

    """

    def add_annotations(
        self,
        img_id: int,
        logits: np.ndarray,
        min_conf: float,
        start_index=1,
    ) -> List[int]:
        """
        add the annotation from the given logits

        Args:
            img_id (int): the id of the image to associate the annotations
            logits (np.ndarray): an array of shape [NClasses, Height, Width]
            min_conf (float): the minimum confidence used to filter the generated masks
            cats_idxs (List, optional): A list that maps the. Defaults to None.
            start_index (int, optional): the index to start generating the coco polygons.
                Normally, 0 encodes the background. Defaults to 1.

        Raises:
            ValueError: if the shape of masks is different than 2
            ValueError: if the shape of probs is different than 3

        Returns:
            List[int]: [the idx of the annotations added]
        """

        if not isinstance(logits, np.ndarray):
            raise ValueError(
                f"the mask type should be a numpy array not a {type(logits)}"
            )

        if len(logits.shape) != 3:
            raise ValueError("masks.shape should equal to 3")

        probs = special.softmax(logits, axis=0)
        masks = probs.argmax(0)

        annotation_ids = []
        for cat_idx in np.unique(masks)[start_index:]:
            mask = (masks == cat_idx).astype(np.uint8)
            conf = np.round(np.nan_to_num(probs[cat_idx, mask].mean()), 2)

            if conf < min_conf:
                continue

            cat_id = int(cat_idx)
            if cat_id not in self.cats:
                raise ValueError(f"cats {cat_id} not in dataset categories")

            groups, n_groups = scipy.ndimage.label(mask)

            # get the groups starting from label 1
            for group_idx in range(1, n_groups + 1):
                group_mask = (groups == group_idx).astype(np.uint8)
                polygons = maskutils.mask_to_polygon(group_mask)
                if len(polygons) == 0:
                    continue

                bbox = maskutils.bbox(polygons, *masks.shape).tolist()
                # an exception is generated when the mask has less than 3 points
                area = int(maskutils.area(group_mask))
                if area == 0:
                    continue
                group_prob_mask = group_mask * probs[cat_idx]
                score = float(np.mean(group_prob_mask[group_prob_mask > 0]))

                annotation_ids.append(
                    self.add_annotation(img_id, cat_id, polygons, area, bbox, 0, score)
                )
        return annotation_ids


@deprecated(version="0.9.34", reason="you should use SemanticCoco")
class SemanticCocoDataset(SemanticCoco):
    def __init__(self, coco_path: str, image_path: str = None) -> None:
        super().__init__(coco_path, image_path)
