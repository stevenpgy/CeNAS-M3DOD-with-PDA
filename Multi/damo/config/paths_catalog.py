# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
            #'img_dir': 'coco/TrainE',
            #'ann_file': 'coco/annotations/instances_trainE.json'
            #'img_dir': 'coco/TrainL',
            #'ann_file': 'coco/annotations/instances_trainK.json'
            #'img_dir': 'coco/TrainH',
            #'ann_file': 'coco/annotations/instances_trainH.json'
        },
        'coco_2017_val': {
            #'img_dir': 'coco/TestCar',
            #'ann_file': 'coco/annotations/instances_testCar.json'
            #'img_dir': 'coco/val2017',
            #'ann_file': 'coco/annotations/instances_val2017.json'
            'img_dir': 'coco/Val',
            'ann_file': 'coco/annotations/instances_val.json'
        },
        'coco_2017_test': {
            'img_dir': 'coco/TestCar',
            'ann_file': 'coco/annotations/instances_testCar2014.json'
            #'img_dir': 'coco/test2017',
            #'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        'coco_test_car': {
            'img_dir': 'coco/TestCar',
            #'img_dir': 'coco/TestCarB',
            'ann_file': 'coco/annotations/instances_testCar2014.json'
        },
        'coco_test_cyc': {
            'img_dir': 'coco/TestCyc',
            #'img_dir': 'coco/TestCycB',
            'ann_file': 'coco/annotations/instances_testCyc2014.json'
        },
        'coco_test_ped': {
            'img_dir': 'coco/TestPed',
            #'img_dir': 'coco/TestPedB',
            'ann_file': 'coco/annotations/instances_testPed2014.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
