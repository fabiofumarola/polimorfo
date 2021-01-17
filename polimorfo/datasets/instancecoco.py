from .coco import CocoDataset
from typing import List
import numpy as np
from ..utils import maskutils

__all__ = ["InstanceCoco"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class InstanceCoco(CocoDataset):
    """
    An extension of the coco dataset to handle the output
        of a instance segmentation models such as MaskRCNN

    """

    def add_annotations(
        self,
        img_id: int,
        labels: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray = None,
    ) -> List[int]:
        """
        add the annotations from the given results

        Args:
            img_id (int): the idx of the image to add the annotations
            labels (np.ndarray): an array with the label predicted
            boxes (np.ndarray): an array of shape (n_labels, 4)
            scores (np.ndarray): an array with the scores of the predictions
            masks (np.ndarray): an array of shape (Height, Width, n_labels) (Optional: None is your saving only boxes)

        Raises:
            ValueError: labels.shape should equal to 1, (n_labels,)
            ValueError: boxes.shape should equal to 2 (n_labels, 4)
            ValueError: masks.shape should equal to 3, (n_labels, Height, Width, n_labels)
            ValueError: scores.shape should equal to 1, (n_labels,)

        Returns:
            List[int]: the idx of the annotations added
        """

        if len(labels.shape) != 1:
            raise ValueError(
                f"labels.shape should equal to 1, (n_labels,) while current has shape {labels.shape}"
            )

        if len(boxes.shape) != 2:
            raise ValueError(
                f"boxes.shape should equal to 2 (n_labels, 4) while current has shape {boxes.shape}"
            )

        if masks is not None:
            masks = masks.squeeze(0)

            if len(masks.shape) != 3:
                raise ValueError(
                    f"masks.shape should equal to 3, (n_labels, Height, Width) while current has shape {masks.shape}"
                )

        if len(scores.shape) != 1:
            raise ValueError("scores.shape should equal to 1, (n_labels,)")

        annotation_ids = []
        for i, cat_id in enumerate(labels):

            if cat_id not in self.cats:
                raise ValueError(f"cats {cat_id} not in dataset categories")
            # convert box to
            x0, y0, x1, y1 = boxes[i]
            w, h = x1 - x0, y1 - y0
            area = int(w * h)
            if area == 0:
                continue

            bbox = [float(x0), float(y0), float(w), float(h)]
            # create the polygons
            if masks is not None:
                mask = masks[i, ...]
                polygons = maskutils.mask_to_polygon(mask)
            else:
                polygons = []

            score = float(scores[i])
            annotation_ids.append(
                self.add_annotation(
                    int(img_id), int(cat_id), polygons, area, bbox, 0, score
                )
            )
        return annotation_ids
