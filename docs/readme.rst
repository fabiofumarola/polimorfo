=========
polimòrfo
=========


.. image:: https://img.shields.io/pypi/v/polimorfo.svg
        :target: https://pypi.python.org/pypi/polimorfo

.. image:: https://img.shields.io/travis/fabiofumarola/polimorfo.svg
        :target: https://travis-ci.com/fabiofumarola/polimorfo

.. image:: https://readthedocs.org/projects/polimorfo/badge/?version=latest
        :target: https://polimorfo.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/fabiofumarola/polimorfo/shield.svg
     :target: https://pyup.io/repos/github/fabiofumarola/polimorfo/
     :alt: Updates



Polimòrfo (πολύμορϕος, comp. di πολυ- «poli-» e μορϕή «forma») is a dataset loader and converter library for object detection segmentation and classification.
The goal of the project is to create a library able to process dataset in format:

.. _COCO: http://cocodataset.org/#format-data
.. _`Pascal VOC`: http://host.robots.ox.ac.uk/pascal/VOC/
.. _`Google Open Images`: https://storage.googleapis.com/openimages/web/download.html

- COCO_: Common Objects in Context
- `Pascal VOC`_: Visual Object Classes Challenge
- `Google Open Images`_: Object Detection and Segmentation dataset released by Google

and transform these dataset into a common format (COCO).

Moreover, the library offers utilies to handle (load, convert, store and transform) the various type of annotations.
This is important when you need to:
- convert mask to polygons
- store mask in a efficient format
- convert mask/poygons into bounding boxes


* Free software: Apache Software License 2.0
* Documentation: https://polimorfo.readthedocs.io.


Features
--------


TODO
=====

- [X] Coco dataset
- [X] download coco datasets for train and val
- [X] add annotations loader and converter
- [X] add the ability to create dataet from scratch
- [ ] add voc dataset format
- [ ]


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
