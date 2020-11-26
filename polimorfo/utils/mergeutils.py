from typing import Union, List, Dict, Any
import numpy as np
from pathlib import Path
from polimorfo.datasets import SemanticCocoDataset, CocoDataset, InstanceCoco
import logging

log = logging.getLogger(__name__)

def rename_annotation_keys(ann: Dict[str, Union[str, int, list, np.ndarray]]) -> Dict:
    """Renames keys for a Coco annotation in order to feed add_annotation method properly.

    Args:
        ann (Dict[str, Union[str, int, list, np.ndarray]]): Coco annotation

    Returns:
        Dict: same as input annotation with keys renamed.
    """
    key_map = { 
                'category_id': 'cat_id',
                'segmentation': 'segmentation',
                'area': 'area',
                'iscrowd': 'is_crowd',
                'score': 'score',
                'bbox': 'bbox'
              }
    assert set(list(ann.keys()) + ['score']) >= (set(key_map.keys())), f'{set(ann.keys())}, {set(key_map.keys())}'
    return { v: ann.get(k, None) for k,v in key_map.items() }

def get_img_meta_by_name(ds, image_name):
    try:
        return [ img_meta for img_meta in ds.imgs.values() if img_meta['file_name'] == image_name ][0]
    except:
        log.warning('image not present in dataset.')
        return None

class AnnotationMerger(object):

    def __init__(self, 
                 root_path: Union[str, Path] = None):
        """Instantiate a merger class for different datasets.

        REMARK: all datasets are expected to point to a common root path for images, but are disjoint sets.
        TODO: extend merger definition.

        REMARK: if root_path is not defined datasets will be merged anyways (since we collect image metadata directly from CocoDataset instances), but you will not be able to visualize images out of the merged output CocoDataset; instead you sould dump the dataset and reload it with appropriate root path.

        Args:
            root_path (Union[str, Path]): [description]
        """
        self.root_path = root_path
        if root_path is not None:
            self.root_path = Path(root_path) 
        else:
            log.info("INFO: root_path is not defined: datasets will be merged anyways, but you will not be able to visualize images; dump the merged dataset and reload it with appropriate root path in case you need it.")
        
    def merge(self, 
              datasets: List[Union[SemanticCocoDataset, CocoDataset, InstanceCoco]],
              dest_path: Union[str, Path] = None
              ) -> CocoDataset:
        
        # compute category to cat_idx map for each dataset
        cat_to_idx = [ { v['name']: k for k,v in ds_item.cats.items() } for ds_item in datasets ] 
        category_names = [ list(ctx.keys()) for ctx in cat_to_idx ]
        flat_category_name = sorted([ item for sublist in category_names for item in sublist ])

        # collect all category names
        merged_cat_names = set(flat_category_name)

        merged_idx_to_cat = dict(enumerate(sorted(merged_cat_names), start=1))
        merged_cat_to_idx = { v:k for k,v in merged_idx_to_cat.items() }

        # map categories for each dataset to merged one
        cat_map = [ { v: merged_cat_to_idx[k]  for k,v in ctx.items() } for ctx in cat_to_idx ]

        # initializize merged dataset
        out_ds = CocoDataset(dest_path, self.root_path) 
        for idx, cat_name in sorted(merged_idx_to_cat.items()):
            out_ds.add_category(cat_name, 'thing')

        for ds_item, cat_map_item in zip(datasets, cat_map):
            # add images
            for img_meta in ds_item.imgs.values():
                # remark: if datasets are not disjoint the might bo overlap here to take care of, i.e. possible multiple definitions of images
                img_idx = out_ds.add_image_from_meta(img_meta)

                orig_img_idx = img_meta['id']
                anns = ds_item.get_annotations(img_idx=orig_img_idx)
                # add anns
                for ann in anns:
                    mapped_ann = rename_annotation_keys(ann)
                    mapped_ann['img_id'] = img_idx # new image idx
                    mapped_ann['cat_id'] = cat_map_item[mapped_ann['cat_id']] # new category index
                    out_ds.add_annotation(**mapped_ann)
                    
        out_ds.reindex()
        return out_ds
