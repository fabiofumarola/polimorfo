=======
History
=======

0.2.0 (2020-02-18)
------------------
* Add support to process coco dataset


0.2.1 (2020-02-28)
------------------
* add support to download files and archives from the web and google drive

0.3.0 (2020-10-04)
-------------------

* addedd support for removing categories and other utilities

0.4.0 (2020-10-05)
-------------------

* addedd support to create a dataset from scratch

0.5.0 (2020-10-06)
-------------------

* added support to visualize images and annotations
* make image removing optional during annotations and categories deletion

0.6.0 (2020-10-12)
-------------------

* added copy dataset
* added split dataset

0.6.1 (2020-10-12)
-------------------

* fixed a bug in colors generation for show images


0.6.2 (2020-10-12)
-------------------

* update signature for function `def update_images_path(self, func):`


0.7.0 (2020-10-19)
-------------------

* add method to dump dataset in format segmentation map


0.8.0 (2020-10-23)
------------------

* fixed bug in maskutils.mask_to_polygons
* add class to transform the predictions from instance and semantic segmentation in coco format
* fixed bug in add_image, add_annotation, add_category
* make load_image and load_images load random images sampled from the dataset

0.8.1 (2020-10-23)
------------------

* fixed bug for tqdm when removing a category and its annotations from the dataset

0.8.2 (2020-10-23)
------------------

* removed the prefix jpg when saving masks
* update draw instance to draw only bounding boxes

0.8.3 (2020-10-24)
------------------
* fixed bug in enum for draw instances


0.8.4 (2020-10-24)
------------------
* add show bounding boxes

0.8.5 (2020-10-24)
------------------
* changed representation for masks from [width, height, labels] to [labels, width, height]

0.8.6 (2020-10-24)
------------------
* added method to crop images
* added method to move annotations with respect a bounding box

0.8.7 (2020-10-24)
------------------
* support fully creation o a new dataset

0.8.8-11 (2020-10-26)
---------------------
* fixed vairous bugs

0.8.12 (2020-10-26)
--------------------
* fixed bug when the size of the segments is equal to 4

0.8.13 (2020-10-26)
--------------------
* fixed bug in json dump to serialize numpy array

0.8.14 (2020-10-26)
--------------------
* fixed bug in json dump to serialize numpy types

0.9.1 (2020-10-28)
--------------------
* fixed various bugs
* add index for speedup lookup operations

0.9.2 (2020-10-28)
---------------------
* add new feature to compute mean average precision and recall per class and global

0.9.3 (2020-10-28)
---------------------
* add computation of mean average precision and mean average recall per image

0.9.4 (2020-10-28)
---------------------
* fixed bug in score computation

0.9.36
-------------------

* fixed bug in mask generation
* feature that allows us to add a single mask per component when saving segmentation results


0.9.38
-----------------

* add min confidence when displaying prediction from a segmentation mask model
* now semantic coco accepts only logits to create annotations


0.9.39
------------

* add new method to remap category idxs


0.9.48
------------

* add new feature to save images and masks to a folder and filter out images and mask with less than k annotations


0.9.52
-----------

* the method `get_segmentation_mask` return also the avg score of the image annotations
* the method `save_mask_images` save also a weight files with the avg score for the image