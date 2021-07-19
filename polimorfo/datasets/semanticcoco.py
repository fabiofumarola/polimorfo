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
            conf = np.round(np.nan_to_num(probs[cat_idx, (mask > 0)].mean()), 2)

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

    def add_annotations_v2(
        self,
        img_id: int,
        probs: np.ndarray,
        min_conf: float,
        start_index=1,
        largest_group_only: bool = False,
    ) -> List[int]:
        """
        add the annotation from the given logits

        Args:
            img_id (int): the id of the image to associate the annotations
            probs (np.ndarray): an array of shape [NClasses, Height, Width] that contains a vector with probabilities
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

        if not isinstance(probs, np.ndarray):
            raise ValueError(
                f"the mask type should be a numpy array not a {type(probs)}"
            )

        if len(probs.shape) != 3:
            raise ValueError("masks.shape should equal to 3")

        masks = probs.argmax(0)

        annotation_ids = []
        for cat_idx in np.unique(masks)[start_index:]:
            mask = (masks == cat_idx).astype(np.uint8)

            cat_id = int(cat_idx)
            if cat_id not in self.cats:
                raise ValueError(f"cats {cat_id} not in dataset categories")

            groups, n_groups = scipy.ndimage.label(mask)
            group_to_consider = list(range(1, n_groups + 1))
            if largest_group_only:
                values, counts = np.unique(groups, return_counts=True)
                group_to_consider = [values[np.argmax(counts[1:]) + 1]]

            # get the groups starting from label 1
            for group_idx in group_to_consider:
                group_mask = (groups == group_idx).astype(np.uint8)
                if group_mask.sum() == 0:
                    continue

                polygons = maskutils.mask_to_polygon(group_mask)
                if len(polygons) == 0:
                    continue

                bbox = maskutils.bbox_from_mask(group_mask)
                if bbox[0] is None:
                    continue

                # FIXME can have problems
                # bbox = maskutils.bbox(polygons, *masks.shape).tolist()
                # an exception is generated when the mask has less than 3 points
                area = int(maskutils.area(group_mask))
                if area == 0:
                    continue
                group_prob_mask = group_mask * probs[cat_idx]
                conf = float(np.median(group_prob_mask[group_prob_mask > 0]))
                if conf < min_conf:
                    continue

                annotation_ids.append(
                    self.add_annotation(img_id, cat_id, polygons, area, bbox, 0, conf)
                )
        return annotation_ids

    def add_annotations_v3(
        self,
        img_id: int,
        probs: np.ndarray,
        min_conf: float,
        single_group: bool = False,
        approx: float = 0.0,
        relative: bool = False,
    ) -> List[int]:
        """Transforms annotations from a probability mask to a coco format

        Args:
            img_id (int): [description]
            probs (np.ndarray): [description]
            min_conf (float): [description]
            start_index (int, optional): [description]. Defaults to 1.
            largest_group_only (bool, optional): [description]. Defaults to False.
            approx (float, optional): the factor used to approximate the polygons by reducint the number of points

        Returns:
            List[int]: [description]
        """
        annotation_ids = []
        global_mask = probs.argmax(0)
        # iterate over the found classes
        for class_idx in np.unique(global_mask):
            if class_idx == 0:
                continue
            # get the probability mask over the class_idx
            class_prob_mask = probs[class_idx] * (global_mask == class_idx)
            # transform the mask to polygons
            class_polygons = maskutils.mask_to_polygon(
                class_prob_mask, min_score=0.5, approx=approx, relative=relative
            )

            if single_group:
                median_conf = np.median(class_prob_mask)
                if median_conf < min_conf:
                    continue

                bbox = maskutils.bbox_from_mask(class_polygons)
                if bbox[0] is None:
                    continue

                area = int(maskutils.area(class_polygons))
                annotation_ids.append(
                    self.add_annotation(
                        img_id, class_idx, class_polygons, area, bbox, 0, median_conf
                    )
                )
            else:
                # for each polyong in polygons
                for poly in class_polygons:
                    poly_mask = maskutils.polygons_to_mask(
                        [poly], global_mask.shape[0], global_mask.shape[1]
                    )
                    poly_mask_prob = poly_mask * probs[class_idx]
                    prob_values = poly_mask_prob[poly_mask_prob > 0]
                    median_conf = float(np.median(prob_values))
                    if median_conf < min_conf:
                        continue

                    bbox = maskutils.bbox_from_mask(poly_mask)
                    if bbox[0] is None:
                        continue
                    area = int(maskutils.area(poly_mask))
                    annotation_ids.append(
                        self.add_annotation(
                            img_id, int(class_idx), [poly], area, bbox, 0, median_conf
                        )
                    )

        return annotation_ids

    def add_annotations_multilabel(
        self,
        img_id: int,
        probs: np.ndarray,
        min_conf: float,
        approx: float = 0.005,
        relative: bool = True,
    ) -> List[int]:
        annotation_ids = []
        for cat_idx in range(len(probs)):
            cat_probs = probs[cat_idx]

            # transform the mask to polygons
            class_polygons = maskutils.mask_to_polygon(
                cat_probs, min_score=0.5, approx=approx, relative=relative
            )

            # for each polyong in polygons
            for poly in class_polygons:

                poly_mask = maskutils.polygons_to_mask(
                    [poly], probs.shape[1], probs.shape[2]
                )
                poly_mask_prob = poly_mask * cat_probs
                prob_values = poly_mask_prob[poly_mask_prob > 0]
                median_conf = float(np.median(prob_values))
                if median_conf < min_conf:
                    continue

                bbox = maskutils.bbox_from_mask(poly_mask)
                if bbox[0] is None:
                    continue

                area = int(maskutils.area(poly_mask))
                annotation_ids.append(
                    self.add_annotation(
                        img_id, int(cat_idx) + 1, [poly], area, bbox, 0, median_conf
                    )
                )

        return annotation_ids


@deprecated(version="0.9.34", reason="you should use SemanticCoco")
class SemanticCocoDataset(SemanticCoco):
    def __init__(self, coco_path: str, image_path: str = None) -> None:
        super().__init__(coco_path, image_path)
