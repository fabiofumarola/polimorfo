.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    import sys
    sys.path.append('..')

.. code:: ipython3

    from polimorfo.datasets import CocoDataset
    from polimorfo.utils import maskutils
    from tqdm import tqdm
    import pandas as pd
    import numpy as np

.. code:: ipython3

    ds_path = '../../car-models/datasets/scratches/val_scratches.json'

.. code:: ipython3

    gt_ds = CocoDataset(ds_path)
    gt_ds.reindex()
    pred_ds = CocoDataset(ds_path)
    pred_ds.reindex()


.. parsed-literal::

    load categories: 100%|██████████| 1/1 [00:00<00:00, 2432.89it/s]
    load images: 100%|██████████| 998/998 [00:00<00:00, 588819.16it/s]
    load annotations: 100%|██████████| 657/657 [00:00<00:00, 571239.16it/s]
    reindex images: 998it [00:00, 768904.37it/s]
    reindex annotations: 657it [00:00, 593508.02it/s]
    load categories: 100%|██████████| 1/1 [00:00<00:00, 3923.58it/s]
    load images: 100%|██████████| 998/998 [00:00<00:00, 695912.78it/s]
    load annotations: 100%|██████████| 657/657 [00:00<00:00, 774931.87it/s]
    reindex images: 998it [00:00, 478171.74it/s]
    reindex annotations: 657it [00:00, 413886.71it/s]


.. code:: ipython3

    gt_ds.remove_annotations([1])
    gt_ds.reindex()


.. parsed-literal::

    reindex images: 998it [00:00, 630884.01it/s]
    reindex annotations: 656it [00:00, 682575.89it/s]
    reindex images: 998it [00:00, 300410.18it/s]
    reindex annotations: 656it [00:00, 532506.95it/s]


.. code:: ipython3

    header = [
            'img_path', 'gt_ann_id', 'pred_ann_id', 'true_class_id',
            'pred_class_id', 'intersection', 'union', 'IOU', 'score'
        ]

.. code:: ipython3

    def best_match(pred_anns, gt_ann_id, gt_mask, img_path, gt_class_id):
        best_pred_ann_id = -1
        best_iou = 0
        best_values = [img_path, gt_ann_id, -1, gt_class_id, 0, 0, 0, 0, 0]
        for pred_ann in pred_anns:
            pred_mask = maskutils.polygons_to_mask(pred_ann['segmentation'], 
                                             gt_img_meta['height'],
                                            gt_img_meta['width'])
            pred_ann_id = pred_ann['id']
            pred_class_id = pred_ann['category_id']
            pred_score = pred_ann['score'] if 'score' in pred_ann else 1
    
            intersection = (pred_mask * gt_mask).sum()
            union = np.count_nonzero(pred_mask + gt_mask)
            iou = intersection / union
            
            if iou > best_iou:
                best_values = [img_path, gt_ann_id, pred_ann_id, gt_class_id,
                               pred_class_id, intersection, union, iou, pred_score]
                best_pred_ann_id = pred_ann_id
                best_iou = iou
        return best_pred_ann_id, best_values

.. code:: ipython3

    results = []
    
    for img_idx, gt_img_meta in tqdm(gt_ds.imgs.items()):
        gt_anns = gt_ds.get_annotations(img_idx)
        pred_img_meta = pred_ds.imgs[img_idx]
        
        if gt_img_meta['file_name'] != pred_img_meta['file_name']:
            raise Exception("images path compared are different")
            
        img_path = gt_img_meta['file_name']
            
        pred_anns = pred_ds.get_annotations(img_idx)
        
        # create a set with all the prediction that will be used to find FP
        pred_idx_dict = {ann['id']: ann for ann in pred_anns}
        
        for gt_ann in gt_anns:
            gt_mask = maskutils.polygons_to_mask(gt_ann['segmentation'], 
                                                 gt_img_meta['height'],
                                                gt_img_meta['width'])
            gt_ann_id = gt_ann['id']
            gt_class_id = gt_ann['category_id']
            
            pred_ann_id, row = best_match(pred_anns, gt_ann_id, gt_mask, img_path, gt_class_id)
            results.append(row)
            if pred_ann_id in pred_idx_dict:
                del pred_idx_dict[pred_ann_id]
                pred_anns = pred_idx_dict.values()
                    
        # false positive dict    
        for pred_ann_id, pred_ann in pred_idx_dict.items():
            results.append([img_path, -1, pred_ann_id, 0,
                                   pred_ann['category_id'], 0, 0, 0, 0])
        


.. parsed-literal::

    100%|██████████| 998/998 [00:01<00:00, 600.94it/s]


.. code:: ipython3

    df = pd.DataFrame(results, columns=header)

.. code:: ipython3

    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>img_path</th>
          <th>gt_ann_id</th>
          <th>pred_ann_id</th>
          <th>true_class_id</th>
          <th>pred_class_id</th>
          <th>intersection</th>
          <th>union</th>
          <th>IOU</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>batch6__2017010400696400__foto0004.jpg</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>968</td>
          <td>968</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>batch6__2017010400696400__foto0004.jpg</td>
          <td>-1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0.0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>batch6__2017010400054500__foto0005.jpg</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>14315</td>
          <td>14315</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>batch6__2017010400885200__foto0001.jpg</td>
          <td>2</td>
          <td>3</td>
          <td>1</td>
          <td>1</td>
          <td>110</td>
          <td>110</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>batch6__2017010400193000__foto0011.jpg</td>
          <td>3</td>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>8267</td>
          <td>8267</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>652</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>651</td>
          <td>652</td>
          <td>1</td>
          <td>1</td>
          <td>2252</td>
          <td>2252</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>653</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>652</td>
          <td>653</td>
          <td>1</td>
          <td>1</td>
          <td>4109</td>
          <td>4109</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>654</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>653</td>
          <td>654</td>
          <td>1</td>
          <td>1</td>
          <td>2657</td>
          <td>2657</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>655</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>654</td>
          <td>655</td>
          <td>1</td>
          <td>1</td>
          <td>2488</td>
          <td>2488</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>656</th>
          <td>batch6__2017010400616100__foto006.jpg</td>
          <td>655</td>
          <td>656</td>
          <td>1</td>
          <td>1</td>
          <td>75990</td>
          <td>75990</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    <p>657 rows × 9 columns</p>
    </div>



.. code:: ipython3

    df['IOU'].mean()




.. parsed-literal::

    0.9984779299847792



.. code:: ipython3

    class_idxs = sorted(df['true_class_id'].unique())[1:]
    for class_idx in class_idxs:
        df_class = df[(df['true_class_id'] == class_idx) | (df['pred_class_id'] == class_idx)]
        

.. code:: ipython3

    df_class




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>img_path</th>
          <th>gt_ann_id</th>
          <th>pred_ann_id</th>
          <th>true_class_id</th>
          <th>pred_class_id</th>
          <th>intersection</th>
          <th>union</th>
          <th>IOU</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>batch6__2017010400696400__foto0004.jpg</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>968</td>
          <td>968</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>batch6__2017010400696400__foto0004.jpg</td>
          <td>-1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0.0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>batch6__2017010400054500__foto0005.jpg</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>14315</td>
          <td>14315</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>batch6__2017010400885200__foto0001.jpg</td>
          <td>2</td>
          <td>3</td>
          <td>1</td>
          <td>1</td>
          <td>110</td>
          <td>110</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>batch6__2017010400193000__foto0011.jpg</td>
          <td>3</td>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>8267</td>
          <td>8267</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>652</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>651</td>
          <td>652</td>
          <td>1</td>
          <td>1</td>
          <td>2252</td>
          <td>2252</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>653</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>652</td>
          <td>653</td>
          <td>1</td>
          <td>1</td>
          <td>4109</td>
          <td>4109</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>654</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>653</td>
          <td>654</td>
          <td>1</td>
          <td>1</td>
          <td>2657</td>
          <td>2657</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>655</th>
          <td>batch6__2017010400901900__foto0004.jpg</td>
          <td>654</td>
          <td>655</td>
          <td>1</td>
          <td>1</td>
          <td>2488</td>
          <td>2488</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>656</th>
          <td>batch6__2017010400616100__foto006.jpg</td>
          <td>655</td>
          <td>656</td>
          <td>1</td>
          <td>1</td>
          <td>75990</td>
          <td>75990</td>
          <td>1.0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    <p>657 rows × 9 columns</p>
    </div>



.. code:: ipython3

    at_iou = 0.5

.. code:: ipython3

    true_positives = df[(df['true_class_id'] == df['pred_class_id']) & (df['IOU'] > at_iou)]
    # all the prediction that do not have a valid gt annotation
    false_positives = df[df['gt_ann_id'] == -1]
    # all the gt annotations that do not have a prediction
    false_negatives = df[df['pred_ann_id'] == -1]

.. code:: ipython3

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    precision




.. parsed-literal::

    0.9984779299847792



.. code:: ipython3

    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    recall




.. parsed-literal::

    1.0



.. code:: ipython3

    a = np.array([-1,0,1,2,3])

.. code:: ipython3

    a[a >0]




.. parsed-literal::

    array([1, 2, 3])



.. code:: ipython3

    df['img_path'].loc[0]




.. parsed-literal::

    'batch6__2017010400696400__foto0004.jpg'



