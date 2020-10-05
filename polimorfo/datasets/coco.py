from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from typing import List
import logging
from PIL import Image
import numpy as np
from datetime import datetime
from typing import Any, List

log = logging.getLogger(__name__)


class UpdateImage():
    """class used to pass a function to change the file_name path
    """

    def __call__(self, in_path: str) -> str:
        pass


class CocoDataset():
    """Process the dataset in COCO format
        Data Format
        ---------
        annotation{
            "id": int, 
            "image_id": int, 
            "category_id": int, 
            "segmentation": RLE or [polygon], 
            "area": float, 
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        }
        categories[{
        "id": int, "name": str, "supercategory": str,
        }]
    """

    def __init__(self, coco_path: str = None, image_path: str = None):
        """Load a dataset from a coco .json dataset
        Arguments:
                        annotations_path {Path} -- Path to coco dataset
        Keyword Arguments:
            images_folder {str} -- the folder wheer the images are saved (default: {'images'})
        """
        self.cats = dict()
        self.imgs = dict()
        self.anns = dict()

        self.cat_id = 1
        self.img_id = 0
        self.ann_id = 0

        self.info = {
            "year": datetime.now().year,
            "version": '1',
            "description": 'dataset create with polimorfo',
            "contributor": '',
            "url": '',
            "date_created": datetime.now().date().isoformat(),
        }

        self.licenses = {}

        if coco_path is not None:
            with Path(coco_path).open() as f:
                data = json.load(f)
            assert set(data) == {
                'annotations', 'categories', 'images', 'info', 'licenses'
            }, 'Not correct file format'

            self.info = data['info']
            self.licenses = data['licenses']

            if image_path is None:
                self.__image_folder = Path(coco_path).parent / 'images'
            else:
                self.__image_folder = Path(image_path)

            for cat_meta in tqdm(data['categories'], desc='load categories'):
                if cat_meta['id'] > self.cat_id:
                    self.cat_id = cat_meta['id']
                self.cats[cat_meta['id']] = cat_meta

            for img_meta in tqdm(data['images'], desc='load images'):
                if img_meta['id'] > self.img_id:
                    self.img_id = img_meta['id']
                self.imgs[img_meta['id']] = img_meta

            for ann_meta in tqdm(data['annotations'], desc='load annotations'):
                if ann_meta['id'] > self.ann_id:
                    self.ann_id = ann_meta['id']
                self.anns[ann_meta['id']] = ann_meta

    def reindex(self):
        """reindex images and annotations to be zero based and categories one based
        """
        old_new_catidx = dict()
        new_cats = dict()
        for new_idx, (old_idx, cat_meta) in enumerate(self.cats.items(), 1):
            old_new_catidx[old_idx] = new_idx
            cat_meta = cat_meta.copy()
            cat_meta['id'] = new_idx
            new_cats[new_idx] = cat_meta
            self.cat_id = new_idx

        old_new_imgidx = dict()
        new_imgs = dict()
        for new_idx, (old_idx, img_meta) in tqdm(enumerate(self.imgs.items()),
                                                 'reindex images'):
            old_new_imgidx[old_idx] = new_idx
            img_meta = img_meta.copy()
            img_meta['id'] = new_idx
            new_imgs[new_idx] = img_meta
            self.img_id = new_idx

        new_anns = dict()
        for new_idx, (old_idx, ann_meta) in tqdm(enumerate(self.anns.items()),
                                                 'reindex annotations'):
            ann_meta = ann_meta.copy()
            ann_meta['id'] = new_idx
            ann_meta['category_id'] = old_new_catidx[ann_meta['category_id']]
            ann_meta['image_id'] = old_new_imgidx[ann_meta['image_id']]
            new_anns[new_idx] = ann_meta
            self.ann_id = new_idx

        del self.cats
        del self.imgs
        del self.anns

        self.cats = new_cats
        self.imgs = new_imgs
        self.anns = new_anns

    def update_image_annotations_path(self, update_images: UpdateImage):
        """update the images path
        Args:
            update_images (UpdateImages): a class with a callable function to change the path
        """

        for img_meta in tqdm(self.imgs.values()):
            img_meta['file_name'] = update_images(img_meta['file_name'])

    def compute_area(self):
        """compute the area of the annotations
        """
        for ann in tqdm(self.anns.values(), desc='process images'):
            ann['area'] = ann['bbox'][2] * ann['bbox'][3]

    def __len__(self):
        """the number of the images in the dataset
        
        Returns:
            [int] -- the number of images in the dataset
        """
        return len(self.imgs)

    def merge_categories(self, cat_to_merge: List[str], new_cat: str):
        """ Merge two or more categories labels to a new single category.
            Remove from __content the category to be merged and update
            annotations cat_ids and reindex data with update content.

        Args:
            cat_to_merge (List[str]): categories to be merged
            new_cat (str): new label to assign to the merged categories
        """
        catidx_to_merge = [
            idx for idx, cat_meta in self.cats.items()
            if cat_meta['name'] in cat_to_merge
        ]
        self.merge_category_ids(catidx_to_merge, new_cat)

    def merge_category_ids(self, cat_to_merge: List[int], new_cat: str):
        """ Merge two or more categories labels to a new single category.
            Remove from __content the category to be merged and update
            annotations cat_ids and reindex data with update content.

        Args:
            cat_to_merge (List[int]): categories to be merged
            new_cat (str): new label to assign to the merged categories
        """
        new_cat_idx = max(self.cats.keys()) + 1

        self.cats = {
            idx: cat
            for idx, cat in self.cats.items()
            if idx not in cat_to_merge
        }
        self.cats[new_cat_idx] = {
            "supercategory": "thing",
            "id": new_cat_idx,
            "name": new_cat
        }

        for ann_meta in tqdm(self.anns.values(), 'process annotations'):
            if ann_meta['category_id'] in cat_to_merge:
                ann_meta['category_id'] = new_cat_idx

        self.reindex()

    def remove_images_without_annotations(self):
        idx_images_with_annotations = {
            ann['image_id'] for ann in self.anns.values()
        }

        idx_to_remove = set(self.imgs.keys()) - idx_images_with_annotations
        for idx in idx_to_remove:
            del self.imgs[idx]
        self.reindex()

    def cleanup_missing_images(self):
        """remove the images missing from images folder
        """
        to_remove_idx = []
        for idx in self.imgs:
            img_meta = self.imgs[idx]
            path = self.__image_folder / img_meta['file_name']
            if not path.exists():
                # There could be paths that have whitespaces renamed (under windows)
                alternative_path = self.__image_folder / img_meta[
                    'file_name'].replace(" ", "_")
                if not alternative_path.exists():
                    del self.imgs[idx]
                    to_remove_idx.append(idx)

        print('removed %d images' % (len(to_remove_idx)))

    def cats_images_count(self):
        """get the number of images per category
        Returns:
            list -- a list of tuples category number of images
        """
        cat_images_dict = defaultdict(set)
        for ann_meta in self.anns.values():
            cat_images_dict[ann_meta['category_id']].add(ann_meta['image_id'])

        return {
            self.cats[idx]['name']: len(images)
            for idx, images in cat_images_dict.items()
        }

    def cats_annotations_count(self):
        """the number of annotations per category
        Returns:
            list -- a list of tuples (category_name, number of annotations)
        """
        cat_anns_dict = defaultdict(set)
        for ann_meta in self.anns.values():
            cat_anns_dict[ann_meta['category_id']].add(ann_meta['id'])

        return {
            self.cats[idx]['name']: len(anns)
            for idx, anns in cat_anns_dict.items()
        }

    def keep_categories(self, ids: List[int]):
        """keep images and annotations only from the selected categories
        Arguments:
            id_categories {list} -- the list of the id categories to keep
        """
        filtered_cat_ids = set(ids)

        self.cats = {
            idx: cat
            for idx, cat in self.cats.items()
            if idx in filtered_cat_ids
        }

        self.anns = {
            idx: ann_meta
            for idx, ann_meta in self.anns.items()
            if ann_meta['category_id'] in filtered_cat_ids
        }

        self.remove_images_without_annotations()

    def remove_images(self, image_idxs: List[int]) -> None:
        """remove all the images and annotations in the specified list

        Arguments:
            image_idxs {List[int]} -- [description]
        """
        image_idxs = set(image_idxs)

        self.imgs = {
            idx: img_meta
            for idx, img_meta in self.imgs.items()
            if idx not in image_idxs
        }

        self.anns = {
            idx: ann_meta
            for idx, ann_meta in self.anns.items()
            if ann_meta['image_id'] not in image_idxs
        }

        catnames_to_remove = {
            cat_name
            for cat_name, count in self.cats_annotations_count().items()
            if count == 0
        }

        self.cats = {
            idx: cat_meta
            for idx, cat_meta in self.cats.items()
            if cat_meta['name'] not in catnames_to_remove
        }

        self.reindex()

    def remove_annotations(self, ids: List[int]) -> None:
        """Remove from the dataset all the annotations ids passes as parameter

        Arguments:
            img_ann_ids {Dict[int, List[Int]]} -- the dictionary of image id annotations ids to remove
        """
        ids = set(ids)
        self.anns = {
            idx: ann for idx, ann in self.anns.items() if idx not in ids
        }
        # remove the images with no annotations
        self.remove_images_without_annotations()
        self.reindex()

    def dumps(self):
        """dump the filtered annotations to a json
        Returns:
            object -- an object with the dumped annotations
        """
        return {
            'info': self.info,
            'licenses': self.licenses,
            'images': list(self.imgs.values()),
            'categories': list(self.cats.values()),
            'annotations': list(self.anns.values()),
        }

    def dump(self, path):
        """dump the dataset annotations and the images to the given path
        Arguments:
            path {str} -- the path to save the json and the images
        """
        with open(path, 'w') as fp:
            json.dump(self.dumps(), fp)

    def load_image(self, idx):
        path = self.__image_folder / self.imgs[idx]['file_name']
        return Image.open(path)

    def mean_pixels(self, sample: int = 1000) -> List[float]:

        channels = {
            'red': 0,
            'green': 0,
            'blue': 0,
        }
        idxs = np.random.choice(list(self.imgs.keys()), sample)

        for idx in tqdm(idxs):
            img = self.load_image(idx)
            for i, color in enumerate(channels.keys()):
                channels[color] += np.mean(img[..., i].flatten())

            del img

        return [
            channels['red'] / sample, channels['green'] / sample,
            channels['blue'] / sample
        ]

    def add_category(self, name: str, supercategory: str) -> int:
        """add a new category to the dataset

        Args:
            name (str): [description]
            supercategory (str): [description]

        Returns:
            int: cat id
        """
        self.cat_id += 1
        self.cats[self.cat_id] = {
            'id': self.cat_id,
            'name': name,
            'supercategory': supercategory
        }
        return self.cat_id

    def add_image(self, image_path: str) -> int:
        """add an image to the dataset

        Args:
            image_path (str): the name of the file

        Returns:
            int: the img id
        """
        self.img_id += 1
        img = Image.open(image_path)
        self.imgs[self.img_id] = {
            'id': self.img_id,
            'width': img.width,
            'height': img.height,
            'filename': Path(image_path).name,
            'flickr_url': '',
            'coco_url': '',
            'data_captured': datetime.now().date().isoformat()
        }
        return self.img_id

    def add_annotation(self, img_id: int, cat_id: int, segmentation: Any,
                       area: float, bbox: List, is_crowd: int) -> int:
        """add a new annotation to the dataset

        Args:
            img_id (int): [description]
            cat_id (int): [description]
            segmentation (Any): [description]
            area (float): [description]
            bbox (List): [description]
            is_crowd (int): [description]

        Returns:
            int: [description]
        """
        assert img_id in self.imgs
        assert cat_id in self.cats

        self.ann_id += 1
        self.anns[self.ann_id] = {
            'id': self.ann_id,
            'image_id': img_id,
            'category_id': cat_id,
            'segmentation': segmentation,
            'area': area,
            'bbox': bbox,
            'iscrowd': is_crowd
        }
        return self.ann_id
