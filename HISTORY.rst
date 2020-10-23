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