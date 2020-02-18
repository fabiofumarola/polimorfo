from pathlib import Path
import json
from tqdm.autonotebook import tqdm
from collections import defaultdict
import logging
from . import utils

log = logging.getLogger(__name__)


class Coco():
    """Process the dataset in COCO format
    """
    def __init__(self, annotations_path, images_folder='images'):
        """Load a dataset from a .json dataset

        Arguments:
            annotations_path {[type]} -- [description]

        Keyword Arguments:
            images_folder {str} -- the folder wheer the images are saved (default: {'images'})
        """
        self.__annotations_path = Path(annotations_path)
        with self.__annotations_path.open() as f:
            self.__content = json.load(f)
        assert set(self.__content.keys()) == {
            'annotations', 'categories', 'images', 'info', 'licenses'
        }, 'Not correct file format'

        self.__image_folder = self.__annotations_path.parent / images_folder
        self.__index_data()

    def __index_data(self):
        self.__imgs = dict()
        self.__imgid_to_anns = defaultdict(list)
        self.__catid_to_imgid = defaultdict(set)
        self.__id_categories = dict()

        for img in tqdm(self.__content['images'], desc='load images'):
            self.__imgs[img['id']] = img

        for ann in tqdm(self.__content['annotations'],
                        desc='load annotations'):
            self.__imgid_to_anns[ann['image_id']].append(ann)
            self.__catid_to_imgid[ann['category_id']].add(ann['image_id'])

        for cat in self.__content['categories']:
            self.__id_categories[cat['id']] = cat

        self.__to_keep_id_categories = {}

    @property
    def to_keep_id_categories(self):
        """return the category ids filtered from the dataset

        Returns:
            set -- the set of the categories idx filtered
        """
        return self.__to_keep_id_categories

    def download_images(self):
        """Download the images from the urls
        """
        self.__image_folder.mkdir(exist_ok=True)
        urls_filepath = [(img['coco_url'],
                          self.__image_folder / img['file_name'])
                         for img in self.__imgs.values()]
        utils.process_images(urls_filepath, 1)
        self.cleanup_missing_images()

    def cleanup_missing_images(self):
        """remove missing images
        """
        count = 0
        for idx, img in self.__imgs.items():
            path = self.__image_folder / img['file_name']
            if not path.exists():
                del self.__imgs[idx]
                count += 1
        print('removed %d images' % (count))

    def get_categories(self, only_name_id=False):
        """return the dataset categories

        Keyword Arguments:
            only_name_id {bool} -- returns only id and category name (default: {False})

        Returns:
            [type] -- [description]
        """
        if only_name_id:
            return {
                idx: cat['name']
                for idx, cat in self.__id_categories.items()
            }
        else:
            return list(self.__id_categories.values())

    def categories_images_count(self):
        """get the number of images per category

        Returns:
            list -- a list of tuples category number of images
        """
        results = list()
        for cat_id, imgs_id in self.__catid_to_imgid.items():
            cat_name = self.__id_categories[cat_id]['name']
            results.append((cat_name, len(imgs_id)))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def categories_annotations_count(self):
        """the number of annotations per category

        Returns:
            list -- a list of tuples (category_name, number of annotations)
        """
        results = list()
        for cat_id, imgs_id in self.__catid_to_imgid.items():
            cat_name = self.__id_categories[cat_id]['name']
            annotations = list()
            for img_id in imgs_id:
                annotations.extend([
                    ann for ann in self.__imgid_to_anns[img_id]
                    if ann['category_id'] == cat_id
                ])
            results.append((cat_name, len(annotations)))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def keep_categories_id(self, id_categories):
        """keep the images only from the selected categories and remove all the annotations

        Arguments:
            id_categories {list} -- the list of the id categories to keep
        """
        self.__to_keep_id_categories = set(id_categories)

    def keep_categories_name(self, name_categories):
        """keep the images only from the selected categories and remove all the annotations

        Arguments:
            id_categories {list} -- the list of the id categories to keep
        """
        results = set()
        name_categories_copy = name_categories.copy()
        for cat in self.__id_categories.values():
            if cat['name'] in name_categories_copy:
                name_categories_copy.remove(cat['name'])
                results.add(cat['id'])
        assert len(results) == len(name_categories), 'wrong name in list'
        self.__to_keep_id_categories = set(results)

    def dumps(self):
        """dump the filtered annotations to a json

        Returns:
            object -- an object with the dumped annotations
        """
        if len(self.__to_keep_id_categories):
            filtered_categories = [
                cat for idx, cat in self.__id_categories.items()
                if idx in self.__to_keep_id_categories
            ]
            filtered_annotations = list()
            filtered_images = list()

            for cat, img_ids in self.__catid_to_imgid.items():
                if cat in self.__to_keep_id_categories:
                    # for each image
                    for img_id in img_ids:
                        # append the image
                        filtered_images.append(self.__imgs[img_id])
                        # filter the annotations wrt the categories selected
                        filtered_img_annotations = [
                            ann for ann in self.__imgid_to_anns[img_id]
                            if ann['category_id'] == cat
                        ]
                        # add the filtered annotations
                        filtered_annotations.extend(filtered_img_annotations)
        else:
            filtered_categories = self.get_categories()
            filtered_images = self.__content['images']
            filtered_annotations = self.__content['annotations']

        return {
            'info': self.__content['info'],
            'licenses': self.__content['licenses'],
            'categories': filtered_categories,
            'annotations': filtered_annotations,
            'images': filtered_images,
        }

    def dump(self, path):
        """dump the dataset annotations and the images to the given path

        Arguments:
            path {str} -- the path to save the json and the images
        """
        data = self.dumps()
        with open(path, 'w') as fp:
            json.dump(data, fp)

    @classmethod
    def train_dataset(cls, folder='.'):
        """Download the Coco annotations and save them to the

        Keyword Arguments:
            folder {str} -- [description] (default: {'.'})
        """
        pass
