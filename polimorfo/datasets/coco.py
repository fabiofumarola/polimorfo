from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from typing import List, Any, Tuple
import logging
from PIL import Image
import numpy as np
from datetime import datetime
from ..utils import maskutils, visualizeutils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy

from enum import Enum

log = logging.getLogger(__name__)

__all__ = ['CocoDataset', 'ExportFormat']


class ExportFormat(Enum):
    coco = 1
    segmentation = 2


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

        # contains the next available id
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

        if image_path is not None:
            self.__image_folder = Path(image_path)

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
            self.cat_id += 1

            for img_meta in tqdm(data['images'], desc='load images'):
                if img_meta['id'] > self.img_id:
                    self.img_id = img_meta['id']
                self.imgs[img_meta['id']] = img_meta
            self.img_id += 1

            for ann_meta in tqdm(data['annotations'], desc='load annotations'):
                if ann_meta['id'] > self.ann_id:
                    self.ann_id = ann_meta['id']
                self.anns[ann_meta['id']] = ann_meta
            self.ann_id += 1

    def copy(self):
        new_coco = CocoDataset(image_path=self.__image_folder)
        new_coco.cats = copy.deepcopy(self.cats)
        new_coco.imgs = copy.deepcopy(self.imgs)
        new_coco.anns = copy.deepcopy(self.anns)
        new_coco.cat_id = self.cat_id
        new_coco.img_id = self.img_id
        new_coco.ann_id = self.ann_id
        new_coco.licenses = self.licenses
        new_coco.info = self.info
        return new_coco

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
        self.cat_id += 1

        old_new_imgidx = dict()
        new_imgs = dict()
        for new_idx, (old_idx, img_meta) in tqdm(enumerate(self.imgs.items()),
                                                 'reindex images'):
            old_new_imgidx[old_idx] = new_idx
            img_meta = img_meta.copy()
            img_meta['id'] = new_idx
            new_imgs[new_idx] = img_meta
            self.img_id = new_idx
        self.img_id += 1

        new_anns = dict()
        for new_idx, (old_idx, ann_meta) in tqdm(enumerate(self.anns.items()),
                                                 'reindex annotations'):
            ann_meta = ann_meta.copy()
            ann_meta['id'] = new_idx
            ann_meta['category_id'] = old_new_catidx[ann_meta['category_id']]
            ann_meta['image_id'] = old_new_imgidx[ann_meta['image_id']]
            new_anns[new_idx] = ann_meta
            self.ann_id = new_idx
        self.ann_id += 1

        del self.cats
        del self.imgs
        del self.anns

        self.cats = new_cats
        self.imgs = new_imgs
        self.anns = new_anns

    def update_images_path(self, func):
        """update the images path
        Args:
            update_images (UpdateImages): a class with a callable function to change the path
        """

        for img_meta in tqdm(self.imgs.values()):
            img_meta['file_name'] = func(img_meta['file_name'])

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

    def remove_categories(self,
                          idxs: List[int],
                          remove_images: bool = False) -> None:
        """Remove the categories with the relative annotations

        Args:
            idxs (List[int]): [description]
        """
        for cat_idx in idxs:
            if cat_idx not in self.cats:
                continue

            for idx in tqdm(list(self.anns), 'process annotations'):
                ann_meta = self.anns[idx]
                if ann_meta['category_id'] == cat_idx:
                    del self.anns[idx]

            del self.cats[cat_idx]

        if remove_images:
            self.remove_images_without_annotations()
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

    def keep_categories(self, ids: List[int], remove_images: bool = False):
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

        if remove_images:
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

    def remove_annotations(self,
                           ids: List[int],
                           remove_images: bool = False) -> None:
        """Remove from the dataset all the annotations ids passes as parameter

        Arguments:
            img_ann_ids {Dict[int, List[Int]]} -- the dictionary of
                image id annotations ids to remove
        """
        ids = set(ids)
        self.anns = {
            idx: ann for idx, ann in self.anns.items() if idx not in ids
        }

        # remove the images with no annotations
        if remove_images:
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

    def dump(self, path, exp_format: ExportFormat = ExportFormat.coco):
        """dump the dataset annotations and the images to the given path

        Args:
            path ([type]): the path to save the json and the images
            exp_format (ExportFormat, optional): [the supported format].
                 Defaults to ExportFormat.coco.

        Raises:
            ValueError: [description]
        """
        path = Path(path)

        if exp_format.value == ExportFormat.coco.value:
            with open(path, 'w') as fp:
                json.dump(self.dumps(), fp)
        elif exp_format.value is ExportFormat.segmentation.value:
            # create a segmentation folder
            if path:
                segments_path = path
            else:
                segments_path = self.__image_folder.parent / 'segments'
            segments_path.mkdir(exist_ok=True, parents=True)

            # save a png of the masks
            for img_idx, img_meta in tqdm(
                    self.imgs.items(),
                    f'savig images in {segments_path.as_posix()}'):
                name = '.'.join(
                    Path(img_meta['file_name']).name.split('.')[:-1])
                segm_path = segments_path / (name + '.png')
                segm_img = self.get_segmentation_mask(img_idx)
                segm_img.save(segm_path)
        else:
            raise ValueError('export format not valid')

    def get_segmentation_mask(self, img_idx: int, cats_idx: List[int] = None):
        """generate a mask for the given image idx

        Args:
            img_idx (int): [the id of the image]
            cats_idx (List[int], optional): [an optional filter over the classes]. Defaults to None.

        Returns:
            [type]: [description]
        """
        img_meta = self.imgs[img_idx]
        width, heigth = img_meta['width'], img_meta['height']
        anns = [ann for ann in self.anns.values() if ann['image_id'] == img_idx]
        if cats_idx:
            anns = [ann for ann in anns if ann['category_id'] in cats_idx]
        segmentations = [obj['segmentation'] for obj in anns]
        if segmentations:
            masks = maskutils.coco_poygons_to_mask(segmentations, heigth, width)
            cats = np.array([obj['category_id'] for obj in anns],
                            dtype=masks.dtype)

            target = np.max((masks * cats[:, None, None]), axis=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 0
        else:
            target = np.zeros((heigth, width), dtype=np.uint8)

        target = Image.fromarray(target)
        return target

    def load_image(self, idx):
        """load an image from the idx

        Args:
            idx ([int]): the idx of the image 

        Returns:
            [Pillow.Image]: []
        """

        path = self.__image_folder / self.imgs[idx]['file_name']
        return Image.open(path)

    def mean_pixels(self, sample: int = 1000) -> List[float]:
        """compute the mean of the pixels

        Args:
            sample (int, optional): [description]. Defaults to 1000.

        Returns:
            List[float]: [description]
        """

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
        self.cats[self.cat_id] = {
            'id': self.cat_id,
            'name': name,
            'supercategory': supercategory
        }
        self.cat_id += 1
        return self.cat_id - 1

    def add_image(self, image_path: str) -> int:
        """add an image to the dataset

        Args:
            image_path (str): the name of the file

        Returns:
            int: the img id
        """
        img = Image.open(image_path)
        self.imgs[self.img_id] = {
            'id': self.img_id,
            'width': img.width,
            'height': img.height,
            'file_name': Path(image_path).name,
            'flickr_url': '',
            'coco_url': '',
            'data_captured': datetime.now().date().isoformat()
        }
        self.img_id += 1
        return self.img_id - 1

    def add_annotation(self,
                       img_id: int,
                       cat_id: int,
                       segmentation: List[List[int]],
                       area: float,
                       bbox: List,
                       is_crowd: int,
                       score: float = None) -> int:
        """add a new annotation to the dataset

        Args:
            img_id (int): [description]
            cat_id (int): [description]
            segmentation (List[List[int]]): [description]
            area (float): [description]
            bbox (List): [description]
            is_crowd (int): [description]
            score (float): [optional score of the prediction]

        Returns:
            int: [description]
        """
        assert img_id in self.imgs
        assert cat_id in self.cats

        metadata = {
            'id': self.ann_id,
            'image_id': img_id,
            'category_id': cat_id,
            'segmentation': segmentation,
            'area': area,
            'bbox': bbox,
            'iscrowd': is_crowd,
        }
        if score:
            metadata['score'] = score

        self.anns[self.ann_id] = metadata
        self.ann_id += 1
        return self.ann_id - 1

    def load_anns(self, ann_idxs):
        if isinstance(ann_idxs, int):
            ann_idxs = [ann_idxs]

        return [self.anns[idx] for idx in ann_idxs]

    def show_image(self,
                   img_idx: int = None,
                   anns_idx: List[int] = None,
                   ax=None,
                   title: str = None,
                   figsize=(18, 6),
                   colors=None) -> plt.Axes:
        """show an image with its annotations

        Args:
            img_idx ([int]): the idx of the image to load (Optional: None)
                in case the value is not specified take a random id
            anns_idx ([List[int]], optional): [description]. Defaults to None.
            ax ([type], optional): [description]. Defaults to None.
            title ([type], optional): [description]. Defaults to None.
            figsize (tuple, optional): [description]. Defaults to (18, 6).

        Returns:
            [plt.Axes]: [description]
        """
        if not img_idx:
            img_idx = np.random.randint(0, self.img_id)

        img = self.load_image(img_idx)

        if anns_idx is None:
            anns = [
                ann for ann in self.anns.values() if ann['image_id'] == img_idx
            ]
        else:
            anns = [self.anns[i] for i in anns_idx]

        boxes = []
        labels = []
        scores = []
        masks = []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            mask = maskutils.polygons_to_mask(ann['segmentation'], img.height,
                                              img.width)
            masks.append(mask)
            if 'score' in ann:
                scores.append(ann['score'])

        if not len(scores):
            scores = [1] * len(anns)

        masks = np.array(masks)
        masks = np.moveaxis(masks, 0, -1)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        class_name_dict = {idx: cat['name'] for idx, cat in self.cats.items()}
        if colors is None:
            colors = visualizeutils.generate_colormap(len(class_name_dict) + 1)

        visualizeutils.draw_instances(img,
                                      boxes,
                                      labels,
                                      scores,
                                      masks,
                                      class_name_dict,
                                      title,
                                      ax=ax,
                                      figsize=figsize,
                                      colors=colors)

        return ax

    def show_images(self,
                    img_idxs: List[int] = None,
                    num_cols=4,
                    figsize=(32, 32)) -> plt.Figure:
        """show the images with their annotations

        Args:
            img_idxs ([List[int]]): a list of image idxs to display (Optional: None)
                If None a random sample of 8 images is taken from the db
            num_cols (int, optional): [description]. Defaults to 4.
            figsize (tuple, optional): [description]. Defaults to (32, 32).

        Returns:
            plt.Figure: [description]
        """
        if not img_idxs:
            img_idxs = np.random.choice(list(self.imgs.keys()), 8,
                                        False).tolist()

        num_rows = len(img_idxs) // num_cols
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.025, hspace=0.05)    # set the spacing between axes.

        class_name_dict = {idx: cat['name'] for idx, cat in self.cats.items()}
        colors = visualizeutils.generate_colormap(len(class_name_dict) + 1)

        for i, img_idx in enumerate(img_idxs):
            ax = plt.subplot(gs[i])
            ax.set_aspect('equal')
            self.show_image(img_idx, ax=ax, colors=colors)

        return fig

    def split(self, train_perc, val_perc, test_perc=None) -> Tuple:
        """split the dataset 

        Args:
            train_perc ([type]): [description]
            val_perc ([type]): [description]
            test_perc ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            Tuple: [description]
        """
        if test_perc is None:
            test_perc = 1 - (train_perc + val_perc)
        if not int(train_perc + val_perc + test_perc) == 1:
            raise ValueError(
                'the sum of train val and test percentage is not equal to 1')

        train_end = int(len(self.imgs) * train_perc)
        val_end = int(len(self.imgs) * (train_perc + val_perc))
        test_perc = int(len(self.imgs) * (train_perc + val_perc + test_perc))

        train_img_ids = list(self.imgs.keys())[:train_end]
        val_img_ids = list(self.imgs.keys())[train_end:val_end]
        test_img_ids = list(self.imgs.keys())[val_end:]

        train_ds = self.copy()
        train_ds.remove_images(val_img_ids + test_img_ids)

        val_ds = self.copy()
        val_ds.remove_images(train_img_ids + test_img_ids)

        test_ds = self.copy()
        test_ds.remove_images(train_img_ids + val_img_ids)

        return train_ds, val_ds, test_ds
