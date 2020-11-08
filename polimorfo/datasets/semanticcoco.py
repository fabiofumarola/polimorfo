from .coco import CocoDataset
from typing import List
import numpy as np
import scipy
from ..utils import maskutils

__all__ = ['SemanticCocoDataset']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SemanticCocoDataset(CocoDataset):
    """
    An extension of the coco dataset to handle the output of a semantic segmentation model

    """

    def add_annotations(self,
                        img_id: int,
                        masks: np.ndarray,
                        probs: np.ndarray,
                        start_index=1) -> List[int]:
        """
        add the annotation from the given masks

        Args:
            img_id (int): the id of the image to associate the annotations
            masks (np.ndarray): a mask of shape [Height, Width]
            probs (np.ndarray): an array of shape [NClasses, Height, Width]
            cats_idxs (List, optional): A list that maps the. Defaults to None.
            start_index (int, optional): the index to start generating the coco polygons.
                Normally, 0 encodes the background. Defaults to 1.

        Raises:
            ValueError: if the shape of masks is different than 2
            ValueError: if the shape of probs is different than 3

        Returns:
            List[int]: [the idx of the annotations added]
        """

        if not isinstance(masks, np.ndarray):
            raise ValueError(
                f'the mask type should be a numpy array not a {type(masks)}')

        if np.count_nonzero(masks) == 0:
            return None

        if len(masks.shape) != 2:
            raise ValueError('masks.shape should equal to 2')

        if len(probs.shape) != 3:
            raise ValueError('masks.shape should equal to 3')

        annotation_ids = []
        # for each class
        for i, class_idx in enumerate(
                np.unique(masks)[start_index:], start_index):
            class_mask = (masks == class_idx).astype(np.uint8)
            class_probs = probs[i]
            cat_id = int(class_idx)

            if cat_id not in self.cats:
                raise ValueError(f'cats {cat_id} not in dataset categories')

            groups, n_groups = scipy.ndimage.label(class_mask)

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
                segment_probs_mask = group_mask * class_probs
                score = float(
                    np.mean(segment_probs_mask[segment_probs_mask > 0]))
                annotation_ids.append(
                    self.add_annotation(img_id, cat_id, polygons, area, bbox, 0,
                                        score))
        return annotation_ids

    def add_annotations_from_scores(self,
                                    img_id: int,
                                    mask_logits: np.ndarray,
                                    start_index=1) -> List[int]:
        """add the annotations from the logit masks

        Args:
            img_id (int): the id of the image to associate the annotations
            mask_logits (np.ndarray): the logits from the semantic model
            start_index (int, optional): the index to start generating the coco polygons.
                Normally, 0 encodes the background. Defaults to 1.

        Returns:
            List[int]: [the idx of the annotations added]]
        """
        masks = np.argmax(mask_logits, axis=0)
        probs = sigmoid(mask_logits)
        return self.add_annotations(img_id, masks, probs, start_index)
