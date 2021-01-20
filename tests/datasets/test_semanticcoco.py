from pathlib import Path

import numpy as np

from polimorfo.datasets.semanticcoco import SemanticCocoDataset

BASE_PATH = Path(__file__).parent.parent / "data"


def test_add_annotations():
    ds = SemanticCocoDataset("fake.json")

    img_id = ds.add_image(BASE_PATH / "test_nodamage.jpg", 100, 100)
    ds.add_category("cat1", "thing")
    ds.add_category("cat2", "thing")

    mask_logits = np.random.rand(3, 256, 256)
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations(img_id, mask_logits, 0.2)

    assert len(ann_idx) > 0


# def test_add_annotations_one_label_per_class():
#     ds = SemanticCocoDataset("fake.json")

#     img_id = ds.add_image(BASE_PATH / "test_nodamage.jpg", 100, 100)
#     ds.add_category("cat1", "thing")
#     ds.add_category("cat2", "thing")

#     mask_logits = np.random.rand(3, 256, 256)
#     mask_logits[1, 0:50, 0:50] = 2

#     ann_idx = ds.add_annotations(img_id, mask_logits, 0.2, one_mask_per_class=True)
#     assert len(ann_idx) > 0


def test_add_annotations_zero_mask():
    ds = SemanticCocoDataset("fake.json")

    img_id = ds.add_image(BASE_PATH / "test_nodamage.jpg", 100, 100)
    ds.add_category("cat1", "thing")
    ds.add_category("cat2", "thing")

    mask_logits = np.zeros((3, 256, 256))
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations(img_id, mask_logits, 0.2)

    assert len(ann_idx) == 0


def test_call_count_statistics():
    ds = SemanticCocoDataset("fake.json")

    img_id = ds.add_image(BASE_PATH / "test_nodamage.jpg", 100, 100)
    ds.add_category("cat1", "thing")
    ds.add_category("cat2", "thing")

    mask_logits = np.random.rand(3, 256, 256)
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations(img_id, mask_logits, 0.4)
    # ds.reindex()

    print(ds.count_annotations_per_category())
    print(ds.count_images_per_category())

    assert len(ann_idx) > 0
