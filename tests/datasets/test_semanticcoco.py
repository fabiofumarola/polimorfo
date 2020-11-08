from polimorfo.datasets.semanticcoco import SemanticCocoDataset
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent / 'data'


def test_add_annotations():
    ds = SemanticCocoDataset('fake.json')

    img_id = ds.add_image(BASE_PATH / 'test_nodamage.jpg')
    ds.add_category('cat1', 'thing')
    ds.add_category('cat2', 'thing')

    mask_logits = np.random.randn(3, 256, 256)
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations_from_scores(img_id, mask_logits)

    assert len(ann_idx) > 0


def test_add_annotations_zero_mask():
    ds = SemanticCocoDataset('fake.json')

    img_id = ds.add_image(BASE_PATH / 'test_nodamage.jpg')
    ds.add_category('cat1', 'thing')
    ds.add_category('cat2', 'thing')

    mask_logits = np.zeros((3, 256, 256))
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations_from_scores(img_id, mask_logits)

    assert len(ann_idx) == 0


def test_call_count_statistics():
    ds = SemanticCocoDataset('fake.json')

    img_id = ds.add_image(BASE_PATH / 'test_nodamage.jpg')
    ds.add_category('cat1', 'thing')
    ds.add_category('cat2', 'thing')

    mask_logits = np.random.randn(3, 256, 256)
    mask_logits[1:3] += 2

    ann_idx = ds.add_annotations_from_scores(img_id, mask_logits)
    # ds.reindex()

    print(ds.count_annotations_per_category())
    print(ds.count_images_per_category())

    assert len(ann_idx) > 0