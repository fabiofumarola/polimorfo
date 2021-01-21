import copy
import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from deprecated import deprecated
from PIL import Image
from tqdm import tqdm

from ..utils import maskutils, visualizeutils

log = logging.getLogger(__name__)

__all__ = ["CocoDataset"]


class CocoDataset:
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

    def __init__(self, coco_path: str, image_path: str = None, verbose: bool = True):
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
        self.img_id = 1
        self.ann_id = 1
        self.index = None
        self.verbose = verbose

        self.info = {
            "year": datetime.now().year,
            "version": "1",
            "description": "dataset create with polimorfo",
            "contributor": "",
            "url": "",
            "date_created": datetime.now().date().isoformat(),
        }

        self.licenses = {}

        self.coco_path = Path(coco_path)

        if image_path is None:
            self.__image_folder = self.coco_path.parent / "images"
        else:
            self.__image_folder = Path(image_path)

        if self.coco_path.exists():
            with self.coco_path.open() as f:
                data = json.load(f)
            assert set(data) == {
                "annotations",
                "categories",
                "images",
                "info",
                "licenses",
            }, "Not correct file format"

            self.info = data["info"]
            self.licenses = data["licenses"]

            for cat_meta in tqdm(
                data["categories"], desc="load categories", disable=not verbose
            ):
                if cat_meta["id"] > self.cat_id:
                    self.cat_id = cat_meta["id"]
                self.cats[cat_meta["id"]] = cat_meta
            self.cat_id += 1

            for img_meta in tqdm(
                data["images"], desc="load images", disable=not verbose
            ):
                if img_meta["id"] > self.img_id:
                    self.img_id = img_meta["id"]
                self.imgs[img_meta["id"]] = img_meta
            self.img_id += 1

            for ann_meta in tqdm(
                data["annotations"], desc="load annotations", disable=not verbose
            ):
                if ann_meta["id"] > self.ann_id:
                    self.ann_id = ann_meta["id"]
                self.anns[ann_meta["id"]] = ann_meta
            self.ann_id += 1

            self.index = Index(self)

    @property
    def images_path(self):
        return self.__image_folder

    def copy(self):
        new_coco = CocoDataset("fake.json", image_path=self.__image_folder.as_posix())
        new_coco.cats = copy.deepcopy(self.cats)
        new_coco.imgs = copy.deepcopy(self.imgs)
        new_coco.anns = copy.deepcopy(self.anns)
        new_coco.cat_id = self.cat_id
        new_coco.img_id = self.img_id
        new_coco.ann_id = self.ann_id
        new_coco.licenses = self.licenses
        new_coco.info = self.info
        new_coco.index = copy.deepcopy(self.index)

        return new_coco

    def reindex(self, by_image_name=True):
        """reindex images and annotations to be zero based and categories one based"""
        old_new_catidx = dict()
        new_cats = dict()
        for new_idx, (old_idx, cat_meta) in enumerate(self.cats.items(), 1):
            old_new_catidx[old_idx] = new_idx
            cat_meta = cat_meta.copy()
            cat_meta["id"] = new_idx
            new_cats[new_idx] = cat_meta
            self.cat_id = new_idx
        self.cat_id += 1

        old_new_imgidx = dict()
        new_imgs = dict()
        if by_image_name:
            sorted_imgs_items = sorted(
                self.imgs.items(), key=lambda x: x[1]["file_name"]
            )
        else:
            sorted_imgs_items = self.imgs.items()

        for new_idx, (old_idx, img_meta) in tqdm(
            enumerate(sorted_imgs_items), "reindex images", disable=not self.verbose
        ):
            old_new_imgidx[old_idx] = new_idx
            img_meta = img_meta.copy()
            img_meta["id"] = new_idx
            new_imgs[new_idx] = img_meta
            self.img_id = new_idx
        self.img_id += 1

        new_anns = dict()
        for new_idx, (old_idx, ann_meta) in tqdm(
            enumerate(self.anns.items()), "reindex annotations"
        ):
            ann_meta = ann_meta.copy()
            ann_meta["id"] = new_idx
            ann_meta["category_id"] = old_new_catidx[ann_meta["category_id"]]
            ann_meta["image_id"] = old_new_imgidx[ann_meta["image_id"]]
            new_anns[new_idx] = ann_meta
            self.ann_id = new_idx
        self.ann_id += 1

        del self.cats
        del self.imgs
        del self.anns

        self.cats = new_cats
        self.imgs = new_imgs
        self.anns = new_anns

        self.index = Index(self)

    def update_images_path(self, func):
        """update the images path
        Args:
            update_images (UpdateImages): a class with a callable function to change the path
        """

        for img_meta in tqdm(self.imgs.values(), disable=not self.verbose):
            img_meta["file_name"] = func(img_meta["file_name"])

    def get_annotations(self, img_idx: int, category_idxs: List[int] = None) -> List:
        """returns the annotations of the given image

        Args:
            img_idx (int): the image idx
            category_idxs (List[int]): the list of the category to filter the returned annotations

        Returns:
            List: a list of the annotations in coco format
        """
        if not self.index:
            self.reindex()

        if category_idxs is None:
            category_idxs = list(self.cats.keys())

        anns_idx = self.index.imgidx_to_annidxs.get(img_idx)
        annotations = []
        for idx in anns_idx:
            ann = self.anns[idx]
            if ann["category_id"] in category_idxs:
                annotations.append(ann)

        return annotations

    def compute_area(self) -> None:
        """compute the area of the annotations"""
        for ann in tqdm(
            self.anns.values(), desc="process images", disable=not self.verbose
        ):
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]

    def __len__(self):
        """the number of the images in the dataset
        Returns:
            [int] -- the number of images in the dataset
        """
        return len(self.imgs)

    def merge_categories(self, cat_to_merge: List[str], new_cat: str) -> None:
        """Merge two or more categories labels to a new single category.
            Remove from __content the category to be merged and update
            annotations cat_ids and reindex data with update content.

        Args:
            cat_to_merge (List[str]): categories to be merged
            new_cat (str): new label to assign to the merged categories
        """
        catidx_to_merge = [
            idx
            for idx, cat_meta in self.cats.items()
            if cat_meta["name"] in cat_to_merge
        ]
        self.merge_category_ids(catidx_to_merge, new_cat)

    def merge_category_ids(self, cat_to_merge: List[int], new_cat: str) -> None:
        """Merge two or more categories labels to a new single category.
            Remove from __content the category to be merged and update
            annotations cat_ids and reindex data with update content.

        Args:
            cat_to_merge (List[int]): categories to be merged
            new_cat (str): new label to assign to the merged categories
        """
        new_cat_idx = max(self.cats.keys()) + 1

        self.cats = {
            idx: cat for idx, cat in self.cats.items() if idx not in cat_to_merge
        }
        self.cats[new_cat_idx] = {
            "supercategory": "thing",
            "id": new_cat_idx,
            "name": new_cat,
        }

        for ann_meta in tqdm(
            self.anns.values(), "process annotations", disable=not self.verbose
        ):
            if ann_meta["category_id"] in cat_to_merge:
                ann_meta["category_id"] = new_cat_idx

        self.reindex()

    def remove_categories(self, idxs: List[int], remove_images: bool = False) -> None:
        """Remove the categories with the relative annotations

        Args:
            idxs (List[int]): [description]
        """
        for cat_idx in idxs:
            if cat_idx not in self.cats:
                continue

            for idx in tqdm(
                list(self.anns), "process annotations", disable=not self.verbose
            ):
                ann_meta = self.anns[idx]
                if ann_meta["category_id"] == cat_idx:
                    del self.anns[idx]

            del self.cats[cat_idx]

        if remove_images:
            self.remove_images_without_annotations()
        self.reindex()

    def remove_images_without_annotations(self):
        idx_images_with_annotations = {ann["image_id"] for ann in self.anns.values()}

        idx_to_remove = set(self.imgs.keys()) - idx_images_with_annotations
        for idx in idx_to_remove:
            del self.imgs[idx]
        self.reindex()

    def cleanup_missing_images(self):
        """remove the images missing from images folder"""
        to_remove_idx = []
        for idx in self.imgs:
            img_meta = self.imgs[idx]
            path = self.__image_folder / img_meta["file_name"]
            if not path.exists():
                # There could be paths that have whitespaces renamed (under windows)
                alternative_path = self.__image_folder / img_meta["file_name"].replace(
                    " ", "_"
                )
                if not alternative_path.exists():
                    del self.imgs[idx]
                    to_remove_idx.append(idx)

        print("removed %d images" % (len(to_remove_idx)))

    def count_images_per_category(self):
        """get the number of images per category
        Returns:
            list -- a list of tuples category number of images
        """
        if not self.index:
            self.reindex()

        return {
            self.cats[cat_id]["name"]: len(set(imgs_list))
            for cat_id, imgs_list in self.index.catidx_to_imgidxs.items()
        }

    def count_annotations_per_category(self):
        """the number of annotations per category
        Returns:
            list -- a list of tuples (category_name, number of annotations)
        """
        if not self.index:
            self.reindex()

        return {
            self.cats[cat_id]["name"]: len(set(anns_list))
            for cat_id, anns_list in self.index.catidx_to_annidxs.items()
        }

    def keep_categories(self, ids: List[int], remove_images: bool = False):
        """keep images and annotations only from the selected categories
        Arguments:
            id_categories {list} -- the list of the id categories to keep
        """
        filtered_cat_ids = set(ids)

        self.cats = {
            idx: cat for idx, cat in self.cats.items() if idx in filtered_cat_ids
        }

        self.anns = {
            idx: ann_meta
            for idx, ann_meta in self.anns.items()
            if ann_meta["category_id"] in filtered_cat_ids
        }

        if remove_images:
            self.remove_images_without_annotations()

    def remove_images(self, image_idxs: List[int]) -> None:
        """remove all the images and annotations in the specified list

        Arguments:
            image_idxs {List[int]} -- [description]
        """
        set_image_idxs = set(image_idxs)

        self.imgs = {
            idx: img_meta
            for idx, img_meta in self.imgs.items()
            if idx not in set_image_idxs
        }

        self.anns = {
            idx: ann_meta
            for idx, ann_meta in self.anns.items()
            if ann_meta["image_id"] not in set_image_idxs
        }

        catnames_to_remove = {
            cat_name
            for cat_name, count in self.count_annotations_per_category().items()
            if count == 0
        }

        self.cats = {
            idx: cat_meta
            for idx, cat_meta in self.cats.items()
            if cat_meta["name"] not in catnames_to_remove
        }

        self.reindex()

    def remove_annotations(self, ids: List[int], remove_images: bool = False) -> None:
        """Remove from the dataset all the annotations ids passes as parameter

        Arguments:
            img_ann_ids {Dict[int, List[Int]]} -- the dictionary of
                image id annotations ids to remove
        """
        set_ids = set(ids)
        self.anns = {idx: ann for idx, ann in self.anns.items() if idx not in set_ids}

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
            "info": self.info,
            "licenses": self.licenses,
            "images": list(self.imgs.values()),
            "categories": list(self.cats.values()),
            "annotations": list(self.anns.values()),
        }

    def dump(self, path=None, **kvargs):
        """dump the dataset annotations and the images to the given path

        Args:
            path ([type]): the path to save the json and the images

        Raises:
            ValueError: [description]
        """
        if path is None:
            path = self.coco_path
        else:
            path = Path(path)

        with open(path, "w") as fp:
            json.dump(self.dumps(), fp)

    def save_idx_class_dict(self, path: Union[str, Path] = None) -> Path:
        """save the idx class dict for the dataset

        Args:
            path (Union[str, Path], optional): [description]. Defaults to None.

        Returns:
            Path: [description]
        """
        if path is None:
            path = self.images_path.parent / "idx_class_dict.json"

        idx_class_dict = {
            str(idx): cat_meta["name"] for idx, cat_meta in self.cats.items()
        }
        with open(path, "w") as f:
            json.dump(idx_class_dict, f)

        return path

    def save_images_and_masks(
        self,
        path: Union[str, Path],
        cats_idx: List[int] = None,
        remapping_dict: Dict[int, int] = None,
        min_conf: float = 0.5,
        ignore_index: int = 255,
        min_num_annotations: int = None,
    ) -> Tuple[Path, Path]:
        """Save images and segmentation mask into folders:
            * segments
            * images
            * weights.csv that contains the pairs image_name, weight
            children of the specified path

        Args:
            path (Union[str, Path], optional): the path to save the masks. Defaults to None.
            cats_idx (List[int], optional): [an optional filter over the classes]. Defaults to None.
            remapping_dict (Dict[int, int], optional): a remapping dictionary for the index to save. Defaults to None.
            min_conf (float): the min confidence to generate the segment, segments with conf below the threshold are replaced as 255
            ignore_index (int): the value used to replace segments with confidence below min_conf
            min_num_annotations (int, optional): [description]. Defaults to None.
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        images_path = path / "images"
        images_path.mkdir(exist_ok=True, parents=True)
        segments_path = path / "segments"
        segments_path.mkdir(exist_ok=True, parents=True)
        scores = []
        scores_path = path / "images_weights.csv"

        for img_idx, img_meta in tqdm(
            self.imgs.items(),
            f"saving masks in {path.as_posix()}",
            disable=not self.verbose,
        ):
            # skip images and mask with less than min_num_annotations
            if (min_num_annotations != None) and (
                len(self.get_annotations(img_idx)) < min_num_annotations
            ):
                continue

            src_img_path = self.__image_folder / img_meta["file_name"]
            dst_imag_path = images_path / img_meta["file_name"]
            if src_img_path.exists():
                shutil.copy(src_img_path, dst_imag_path)

            name = ".".join(Path(img_meta["file_name"]).name.split(".")[:-1])
            segm_path = segments_path / (name + ".png")
            if segm_path.exists():
                continue
            segm_img, avg_score = self.get_segmentation_mask(
                img_idx, cats_idx, remapping_dict, min_conf, ignore_index
            )
            segm_img.save(segm_path)
            scores.append(f"{segm_path.name},{avg_score}\n")

        cat_idx_dict = dict()
        for idx, cat in self.cats.items():
            cat_idx_dict[cat["name"]] = idx

        with open(scores_path, "w") as f:
            f.writelines(scores)

        with open(path / "cat_idx_dict.json", "w") as f:
            json.dump(cat_idx_dict, f)

        return images_path, segments_path

    def save_segmentation_masks(
        self,
        path: Union[str, Path] = None,
        cats_idx: List[int] = None,
        remapping_dict: Dict[int, int] = None,
        min_conf: float = 0.5,
        ignore_index: int = 255,
    ) -> None:
        """save the segmentation mask for the given dataset

        Args:
            path (Union[str, Path], optional): the path to save the masks. Defaults to None.
            cats_idx (List[int], optional): [an optional filter over the classes]. Defaults to None.
            remapping_dict (Dict[int, int], optional): a remapping dictionary for the index to save. Defaults to None.
            min_conf (float): the min confidence to generate the segment, segments with conf below the threshold are replaced as 255
            ignore_index (int): the value used to replace segments with confidence below min_conf
        """
        if path is None:
            path = self.__image_folder.parent / "segments"
        else:
            path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        for img_idx, img_meta in tqdm(
            self.imgs.items(),
            f"saving masks in {path.as_posix()}",
            disable=not self.verbose,
        ):
            name = ".".join(Path(img_meta["file_name"]).name.split(".")[:-1])
            segm_path = path / (name + ".png")
            if segm_path.exists():
                continue
            segm_img, _ = self.get_segmentation_mask(
                img_idx, cats_idx, remapping_dict, min_conf, ignore_index
            )
            segm_img.save(segm_path)

        cat_idx_dict = dict()
        for idx, cat in self.cats.items():
            cat_idx_dict[cat["name"]] = idx

        with open(path.parent / "cat_idx_dict.json", "wa") as f:
            json.dump(cat_idx_dict, f)

    def remap_categories(self, remapping_dict: Dict[int, int]) -> None:
        for ann in tqdm(self.anns.values(), desc="renaming category idxs"):
            if ann["category_id"] in remapping_dict:
                ann["category_id"] = remapping_dict[ann["category_id"]]

        cats = dict()
        for idx, cat in self.cats.items():
            if idx in remapping_dict:
                new_idx = remapping_dict[idx]
            else:
                new_idx = idx
            cat["id"] = new_idx
            cats[new_idx] = cat

        self.cats = cats
        self.cats = dict(sorted(self.cats.items(), key=lambda x: x[0]))
        self.index = Index(self)

    def get_segmentation_mask(
        self,
        img_idx: int,
        cats_idx: List[int] = None,
        remapping_dict: Dict[int, int] = None,
        min_conf: float = 0.5,
        ignore_index: int = 255,
    ) -> Tuple[Image.Image, float]:
        """generate a mask and weight for the given image idx

        Args:
            img_idx (int): [the id of the image]
            cats_idx (List[int], optional): [an optional filter over the classes]. Defaults to None.
            remapping_dict (Dict[int, int], optional): [description]. Defaults to None.
            min_conf (float): the min confidence to generate the segment, segments with conf below the threshold are replaced as 255
            ignore_index (int): the value used to replace segments with confidence below min_conf

        Returns:
            Tuple[Image.Image, float]: [description]
        """
        img_meta = self.imgs[img_idx]
        height, width = img_meta["height"], img_meta["width"]
        anns = self.get_annotations(img_idx, cats_idx)
        target_image = np.zeros((height, width), dtype=np.uint8)
        score = 0
        count = 0

        segmentations = [obj["segmentation"] for obj in anns]
        if len(segmentations):
            annotation_masks = maskutils.coco_poygons_to_mask(
                segmentations, height, width
            )
            elements = []
            for i, obj in enumerate(anns):
                elements.append(
                    {
                        "id": obj["category_id"],
                        "area": obj["area"],
                        "mask": annotation_masks[i],
                        "score": obj["score"] if "score" in obj else 1.0,
                    }
                )
            # order the mask by area
            elements = sorted(elements, key=lambda x: x["area"], reverse=True)
            for elem in elements:
                if elem["score"] < min_conf:
                    target_image[elem["mask"] == 1] = ignore_index
                else:
                    score += elem["score"]
                    count += 1
                    if remapping_dict is not None and elem["id"] in remapping_dict:
                        target_image[elem["mask"] == 1] = remapping_dict[elem["id"]]
                    else:
                        target_image[elem["mask"] == 1] = elem["id"]

        target = Image.fromarray(target_image)
        avg_score = score / count if count else count
        return target, avg_score

    def load_image(self, idx):
        """load an image from the idx

        Args:
            idx ([int]): the idx of the image

        Returns:
            [Pillow.Image]: []
        """

        path = self.__image_folder / self.imgs[idx]["file_name"]
        return Image.open(path)

    def mean_pixels(self, sample: int = 1000) -> List[float]:
        """compute the mean of the pixels

        Args:
            sample (int, optional): [description]. Defaults to 1000.

        Returns:
            List[float]: [description]
        """

        channels = {
            "red": 0,
            "green": 0,
            "blue": 0,
        }
        idxs = np.random.choice(list(self.imgs.keys()), sample)

        for idx in tqdm(idxs, disable=not self.verbose):
            img = np.array(self.load_image(idx))
            for i, color in enumerate(channels.keys()):
                channels[color] += np.mean(img[..., i].flatten())

            del img

        return [
            channels["red"] / sample,
            channels["green"] / sample,
            channels["blue"] / sample,
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
            "id": self.cat_id,
            "name": name,
            "supercategory": supercategory,
        }
        self.cat_id += 1
        return self.cat_id - 1

    def add_image(
        self, file_name: Union[str, Path], height: int, width: int, **kwargs
    ) -> int:
        """Add a new image to the dataset .

        Args:
            file_name (Union[str, Path]): the file name holding the image
            height (int): the height of the image
            width (int): the width of the image

        Returns:
            int: [description]
        """
        if isinstance(file_name, Path):
            file_name = file_name.as_posix()

        self.imgs[self.img_id] = {
            "id": self.img_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "flickr_url": "",
            "coco_url": "",
            "data_captured": datetime.now().date().isoformat(),
        }
        self.img_id += 1
        return self.img_id - 1

    def add_annotation(
        self,
        img_id: int,
        cat_id: int,
        segmentation: List[List[int]],
        area: float,
        bbox: List,
        is_crowd: int,
        score: float = None,
    ) -> int:
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
            "id": self.ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": is_crowd,
        }
        if score:
            metadata["score"] = score

        self.anns[self.ann_id] = metadata
        self.ann_id += 1
        return self.ann_id - 1

    def crop_image(
        self, img_idx: int, bbox: Tuple[float, float, float, float], dst_path: Path
    ) -> str:
        """crop the image id with respect the given bounding box to the specified path

        Args:
            img_idx (int): the id of the image
            bbox (Tuple[float, float, float, float]): a bounding box with the format [Xmin, Ymin, Xmax, Ymax]
            dst_path (Path): the path where the image has to be saved

        Returns:
            str: the name of the image
        """
        dst_path = Path(dst_path)
        img_meta = self.imgs[img_idx]
        img = self.load_image(img_idx)
        img_cropped = img.crop(bbox)
        img_cropped.save(dst_path / img_meta["file_name"])
        return img_meta["file_name"]

    def enlarge_box(self, bbox, height, width, pxls=10):
        """enlarge a given box of pxls pixels

        Args:
            bbox ([type]): a tuple, list of np.arry of shape (4,)
            height (int): the height of the image
            width (int): the width of the image
            pxls (int, optional): the number of pixels to add. Defaults to 10.

        Returns:
            boundingbox: the enlarged bounding box
        """
        bbox = bbox.copy()
        bbox[0] = np.clip(bbox[0] - pxls, 0, width)
        bbox[1] = np.clip(bbox[1] - pxls, 0, height)
        bbox[2] = np.clip(bbox[2] + pxls, 0, width)
        bbox[3] = np.clip(bbox[3] + pxls, 0, height)
        return bbox

    def move_annotation(
        self, idx: int, bbox: Tuple[float, float, float, float]
    ) -> Dict:
        """move the bounding box and the segments of the annotation with respect to given bounding box

        Args:
            idx (int): the annotation idx
            bbox (Tuple[float, float, float, float]): the bounding box

        Returns:
            Dict: a dictioary with the keys iscrowd, bboox, area, segmentation
        """

        ann_meta = self.anns[idx]
        img_meta = self.imgs[ann_meta["image_id"]]
        img_bbox = np.array([0, 0, img_meta["width"], img_meta["height"]])

        # compute the shift for x and y
        diff_bbox = img_bbox - np.array(bbox)
        move_width, move_height = diff_bbox[:2]

        # move bbox
        bbox_moved = copy.deepcopy(ann_meta["bbox"])
        bbox_moved[0] += move_width
        bbox_moved[1] += move_height

        # move segmentations
        segmentations_moved = copy.deepcopy(ann_meta["segmentation"])
        for segmentation in segmentations_moved:
            for i in range(len(segmentation)):
                if i % 2 == 0:
                    segmentation[i] += move_width
                else:
                    segmentation[i] += move_height

        ann_meta_moved = {
            "iscrowd": ann_meta["iscrowd"],
            "bbox": bbox_moved,
            "area": ann_meta["area"],
            "segmentation": segmentations_moved,
        }

        return ann_meta_moved

    def load_anns(self, ann_idxs):
        if isinstance(ann_idxs, int):
            ann_idxs = [ann_idxs]

        return [self.anns[idx] for idx in ann_idxs]

    def show_image(
        self,
        img_idx: int = None,
        img_name: str = None,
        anns_idx: List[int] = None,
        ax=None,
        title: str = None,
        figsize=(18, 6),
        colors=None,
        show_boxes=False,
        show_masks=True,
        min_score=0.5,
        min_area: int = 0,
        cats_idx: List[int] = None,
    ) -> plt.Axes:
        """show an image with its annotations

        Args:
            img_idx (int, optional): the idx of the image to load (Optional: None)
                in case the value is not specified take a random id
            img_name (str, optional): the name of the image to load
            anns_idx (List[int], optional): [description]. Defaults to None.
            ax ([type], optional): [description]. Defaults to None.
            title (str, optional): [description]. Defaults to None.
            figsize (tuple, optional): [description]. Defaults to (18, 6).
            colors ([type], optional): [description]. Defaults to None.
            show_boxes (bool, optional): [description]. Defaults to False.
            show_masks (bool, optional): [description]. Defaults to True.
            min_score (float, optional): [description]. Defaults to 0.5.
            cats_idx (List, optional): the list of categories to show. Defaults to None to display all the categories

        Returns:
            plt.Axes: [description]
        """
        if img_idx is None:
            img_idx = np.random.randint(0, self.img_id)

        if img_name is not None:
            values = [
                idx
                for idx, img_meta in self.imgs.items()
                if img_meta["file_name"] == img_name
            ]
            img_idx = img_idx if len(values) == 0 else values[0]

        img = self.load_image(img_idx)

        if title is None:
            title = self.imgs[img_idx]["file_name"]

        if anns_idx is None:
            anns_idx = self.index.imgidx_to_annidxs[img_idx]
        anns = [self.anns[i] for i in anns_idx]

        if cats_idx is not None:
            anns = [ann for ann in anns if ann["category_id"] in cats_idx]

        boxes = []
        labels = []
        scores = []
        masks = []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            if "segmentation" in ann:
                mask = maskutils.polygons_to_mask(
                    ann["segmentation"], img.height, img.width
                )
                masks.append(mask)
            if "score" in ann:
                scores.append(float(ann["score"]))

        if not len(scores):
            scores = [1] * len(anns)

        if len(masks):
            masks = np.array(masks)
        else:
            masks = None

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)

        idx_class_dict = {idx: cat["name"] for idx, cat in self.cats.items()}
        if colors is None:
            colors = visualizeutils.generate_colormap(len(idx_class_dict) + 1)

        visualizeutils.draw_instances(
            img,
            boxes,
            labels,
            scores,
            masks,
            idx_class_dict,
            title,
            ax=ax,
            figsize=figsize,
            colors=colors,
            show_boxes=show_boxes,
            show_masks=show_masks,
            min_score=min_score,
            min_area=min_area,
            box_type=visualizeutils.BoxType.xywh,
        )

        return ax

    def show_images(
        self,
        idxs_or_num: Union[List[int], int] = None,
        num_cols=4,
        figsize=(32, 32),
        show_masks=True,
        show_boxes=False,
        min_score: float = 0.5,
        min_area: int = 0,
        cats_idx: List[int] = None,
    ) -> plt.Figure:
        """show the images with their annotations

        Args:
            img_idxs (Union[List[int], int]): a list of image idxs to display or the number of images (Optional: None)
                If None a random sample of 8 images is taken from the db
            num_cols (int, optional): [description]. Defaults to 4.
            figsize (tuple, optional): [description]. Defaults to (32, 32).
            show_masks (bool, optional): [description]. Defaults to True.
            show_boxes (bool, optional): [description]. Defaults to False.
            min_score (float, optional): [description]. Defaults to 0.5.
            min_area (int, optional): the min area of the annotations to display, Default to 0
        Returns:
            plt.Figure: [description]
        """
        if idxs_or_num is None:
            img_idxs = np.random.choice(list(self.imgs.keys()), 8, False).tolist()
        elif isinstance(idxs_or_num, int):
            img_idxs = np.random.choice(
                list(self.imgs.keys()), idxs_or_num, False
            ).tolist()
        else:
            img_idxs = idxs_or_num

        num_rows = len(img_idxs) // num_cols
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)
        gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

        class_name_dict = {idx: cat["name"] for idx, cat in self.cats.items()}
        colors = visualizeutils.generate_colormap(len(class_name_dict) + 1)

        for i, img_idx in enumerate(img_idxs):
            ax = plt.subplot(gs[i])
            ax.set_aspect("equal")
            self.show_image(
                img_idx,
                ax=ax,
                colors=colors,
                show_masks=show_masks,
                show_boxes=show_boxes,
                min_score=min_score,
                min_area=min_area,
                cats_idx=cats_idx,
            )

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
                "the sum of train val and test percentage is not equal to 1"
            )

        train_end = int(len(self.imgs) * train_perc)
        val_end = int(len(self.imgs) * (train_perc + val_perc))
        test_perc = int(len(self.imgs) * (train_perc + val_perc + test_perc))

        train_img_ids = list(self.imgs.keys())[:train_end]
        val_img_ids = list(self.imgs.keys())[train_end:val_end]
        test_img_ids = list(self.imgs.keys())[val_end:]

        train_ds = self.copy()
        train_ds.remove_images(val_img_ids + test_img_ids)
        train_ds.reindex()

        val_ds = self.copy()
        val_ds.remove_images(train_img_ids + test_img_ids)
        train_ds.reindex()

        test_ds = self.copy()
        test_ds.remove_images(train_img_ids + val_img_ids)
        train_ds.reindex()

        return train_ds, val_ds, test_ds


class Index(object):
    def __init__(self, coco: CocoDataset) -> None:
        self.catidx_to_imgidxs: DefaultDict[int, Set[int]] = defaultdict(set)
        self.imgidx_to_annidxs: DefaultDict[int, Set[int]] = defaultdict(set)
        self.catidx_to_annidxs: DefaultDict[int, Set[int]] = defaultdict(set)

        for img_idx in coco.imgs.keys():
            self.imgidx_to_annidxs[img_idx] = set()

        for idx, ann_meta in coco.anns.items():
            self.catidx_to_imgidxs[ann_meta["category_id"]].add((ann_meta["image_id"]))
            self.imgidx_to_annidxs[ann_meta["image_id"]].add((idx))
            self.catidx_to_annidxs[ann_meta["category_id"]].add(idx)
