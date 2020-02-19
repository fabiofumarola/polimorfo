.. code:: ipython3

    import sys
    sys.path.append('../')

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    from polimorfo.datasets.coco import Coco
    import json


.. parsed-literal::

    /Users/fumarolaf/miniconda3/envs/an/lib/python3.7/site-packages/tqdm/autonotebook/__init__.py:14: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
      " (e.g. in jupyter console)", TqdmExperimentalWarning)


.. code:: ipython3

    coco = Coco('./annotations/instances_train2017.json')



.. parsed-literal::

    HBox(children=(IntProgress(value=0, description='load images', max=118287, style=ProgressStyle(description_wid…


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, description='load annotations', max=860001, style=ProgressStyle(descriptio…


.. parsed-literal::

    


.. code:: ipython3

    print(coco.get_categories(True))


.. parsed-literal::

    {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


.. code:: ipython3

    print(coco.categories_images_count())


.. parsed-literal::

    [('person', 64115), ('chair', 12774), ('car', 12251), ('dining table', 11837), ('cup', 9189), ('bottle', 8501), ('bowl', 7111), ('handbag', 6841), ('truck', 6127), ('bench', 5570), ('backpack', 5528), ('book', 5332), ('cell phone', 4803), ('sink', 4678), ('clock', 4659), ('tv', 4561), ('potted plant', 4452), ('couch', 4423), ('dog', 4385), ('knife', 4326), ('sports ball', 4262), ('traffic light', 4139), ('cat', 4114), ('umbrella', 3968), ('bus', 3952), ('tie', 3810), ('bed', 3682), ('vase', 3593), ('train', 3588), ('fork', 3555), ('spoon', 3529), ('laptop', 3524), ('motorcycle', 3502), ('surfboard', 3486), ('skateboard', 3476), ('tennis racket', 3394), ('toilet', 3353), ('bicycle', 3252), ('bird', 3237), ('pizza', 3166), ('skis', 3082), ('remote', 3076), ('boat', 3025), ('airplane', 2986), ('horse', 2941), ('cake', 2925), ('oven', 2877), ('baseball glove', 2629), ('giraffe', 2546), ('wine glass', 2533), ('baseball bat', 2506), ('suitcase', 2402), ('sandwich', 2365), ('refrigerator', 2360), ('kite', 2261), ('banana', 2243), ('frisbee', 2184), ('elephant', 2143), ('teddy bear', 2140), ('keyboard', 2115), ('cow', 1968), ('broccoli', 1939), ('zebra', 1916), ('mouse', 1876), ('stop sign', 1734), ('fire hydrant', 1711), ('orange', 1699), ('carrot', 1683), ('snowboard', 1654), ('apple', 1586), ('microwave', 1547), ('sheep', 1529), ('donut', 1523), ('hot dog', 1222), ('toothbrush', 1007), ('bear', 960), ('scissors', 947), ('parking meter', 705), ('toaster', 217), ('hair drier', 189)]


.. code:: ipython3

    print(coco.categories_annotations_count())


.. parsed-literal::

    [('person', 262465), ('car', 43867), ('chair', 38491), ('book', 24715), ('bottle', 24342), ('cup', 20650), ('dining table', 15714), ('bowl', 14358), ('traffic light', 12884), ('handbag', 12354), ('umbrella', 11431), ('bird', 10806), ('boat', 10759), ('truck', 9973), ('bench', 9838), ('sheep', 9509), ('banana', 9458), ('kite', 9076), ('motorcycle', 8725), ('backpack', 8720), ('potted plant', 8652), ('cow', 8147), ('wine glass', 7913), ('carrot', 7852), ('knife', 7770), ('broccoli', 7308), ('donut', 7179), ('bicycle', 7113), ('skis', 6646), ('vase', 6613), ('horse', 6587), ('tie', 6496), ('cell phone', 6434), ('orange', 6399), ('cake', 6353), ('sports ball', 6347), ('clock', 6334), ('suitcase', 6192), ('spoon', 6165), ('surfboard', 6126), ('bus', 6069), ('apple', 5851), ('pizza', 5821), ('tv', 5805), ('couch', 5779), ('remote', 5703), ('sink', 5610), ('skateboard', 5543), ('elephant', 5513), ('dog', 5508), ('fork', 5479), ('zebra', 5303), ('airplane', 5135), ('giraffe', 5131), ('laptop', 4970), ('tennis racket', 4812), ('teddy bear', 4793), ('cat', 4768), ('train', 4571), ('sandwich', 4373), ('bed', 4192), ('toilet', 4157), ('baseball glove', 3747), ('oven', 3334), ('baseball bat', 3276), ('hot dog', 2918), ('keyboard', 2855), ('snowboard', 2685), ('frisbee', 2682), ('refrigerator', 2637), ('mouse', 2262), ('stop sign', 1983), ('toothbrush', 1954), ('fire hydrant', 1865), ('microwave', 1673), ('scissors', 1481), ('bear', 1294), ('parking meter', 1285), ('toaster', 225), ('hair drier', 198)]


.. code:: ipython3

    coco.keep_categories_id([89,3,4])

.. code:: ipython3

    coco.to_keep_id_categories




.. parsed-literal::

    {3, 4, 89}



.. code:: ipython3

    coco.keep_categories_name(['hair drier', 'toaster', 'bear'])

.. code:: ipython3

    coco.to_keep_id_categories




.. parsed-literal::

    {23, 80, 89}



.. code:: ipython3

    res = coco.dumps()

.. code:: ipython3

    res.keys()




.. parsed-literal::

    dict_keys(['info', 'licenses', 'categories', 'annotations', 'images'])



.. code:: ipython3

    print('images', len(res['images']))
    print('annotations', len(res['annotations']))


.. parsed-literal::

    images 1366
    annotations 1717


.. code:: ipython3

    coco.dump('hair_drier_toaster_bear.json')

.. code:: ipython3

    small_dataset = Coco('hair_drier_toaster_bear.json')



.. parsed-literal::

    HBox(children=(IntProgress(value=0, description='load images', max=1366, style=ProgressStyle(description_width…


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, description='load annotations', max=1717, style=ProgressStyle(description_…


.. parsed-literal::

    


.. code:: ipython3

    small_dataset.download_images()



.. parsed-literal::

    HBox(children=(IntProgress(value=0, description='download images', max=1366, style=ProgressStyle(description_w…


.. parsed-literal::

    
    removed 0 images

