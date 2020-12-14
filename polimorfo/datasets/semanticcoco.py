from typing import List

import numpy as np
import scipy
from deprecated import deprecated

from ..utils import maskutils
from .coco import CocoDataset

__all__ = ['SemanticCoco', 'SemanticCocoDataset']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SemanticCoco(CocoDataset):
    """
    An extension of the coco dataset to handle the output of a semantic segmentation model

    """

    def add_annotations(self,
                        img_id: int,
                        masks: np.ndarray,
                        probs: np.ndarray,
                        min_score: float = 0.5,
                        one_mask_per_class: bool = False,
                        start_index=1) -> List[int]:
        """
        add the annotation from the given masks

        Args:
            img_id (int): the id of the image to associate the annotations
            masks (np.ndarray): a mask of shape [Height, Width]
            probs (np.ndarray): an array of shape [NClasses, Height, Width]
            min_score (float, optional): the minimum score to use to filter the generated masks
            one_mask_per_class (bool, optional): if True save only the largest mask per class (default: False)
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

        #zeroing all the pixel with confidence lower than min_score
        # masks[probs < min_score] = 0
        probs[probs < min_score] = 0

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

            if one_mask_per_class:
                largest_area = 0
                largest_id = 0

                for group_idx in range(1, n_groups + 1):
                    group_mask = (groups == group_idx).astype(np.uint8)
                    if group_mask.sum() > largest_area:
                        largest_area = group_mask.sum()
                        largest_id = group_idx

                n_groups = 1
                groups[groups != largest_id] = 0

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
                                    min_score: float = 0.5,
                                    one_mask_per_class: bool = False,
                                    start_index=1) -> List[int]:
        """add the annotations from the logit masks

        Args:
            img_id (int): the id of the image to associate the annotations
            mask_logits (np.ndarray): the logits from the semantic model
            min_score (float, optional): the minimum score to use to filter the generated masks
            one_mask_per_class (bool, optional): if True save only the largest mask per class (default: False)
            start_index (int, optional): the index to start generating the coco polygons.
                Normally, 0 encodes the background. Defaults to 1.

        Returns:
            List[int]: [the idx of the annotations added]]
        """
        masks = np.argmax(mask_logits, axis=0)
        probs = sigmoid(mask_logits)
        # put to zero all the score lower than min_score
        probs[probs < min_score] = 0
        return self.add_annotations(img_id, masks, probs, 0, one_mask_per_class,
                                    start_index)


@deprecated(version='0.9.34', reason='you should use SemanticCoco')
class SemanticCocoDataset(SemanticCoco):

    def __init__(self, coco_path: str, image_path: str = None) -> None:
        super().__init__(coco_path, image_path)
