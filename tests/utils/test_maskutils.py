import numpy as np
import pytest

from polimorfo.utils import maskutils


def test_mask_to_polygons():
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    polygons = maskutils.mask_to_polygon(mask)
    assert len(polygons) == 1


def test_bbox_dimension():
    mask = np.zeros((15, 16))
    mask[4:10, 5:12] = 1
    polygons = maskutils.mask_to_polygon(mask, approx=0)
    bbox = maskutils.bbox(polygons, mask.shape[0], mask.shape[1])
    assert tuple(bbox.tolist()) == (5, 4, 6, 5)


def test_bbox():
    polygons = [[1, 2, 2, 3, 5, 5, 10, 25]]
    bbox = maskutils.bbox(polygons, 64, 64)
    area = maskutils.area(bbox)
    assert area == 4


def test_bbox_zero_area():
    polygons = [[1, 1, 2, 2, 3, 3, 4, 4]]
    bbox = maskutils.bbox(polygons, 64, 64)
    area = maskutils.area(bbox)
    assert area == 0


def test_bbox_invalid_polygons():
    with pytest.raises(Exception) as ex:
        polygons = [[]]
        maskutils.bbox(polygons, 64, 64)
        assert ex.value == "input type is not supported."

    with pytest.raises(Exception) as ex:
        polygons = [[1, 1, 2, 2]]
        maskutils.bbox(polygons, 64, 64)
        assert (
            ex.value
            == "Argument 'bb' has incorrect type (expected numpy.ndarray, got list)"
        )
