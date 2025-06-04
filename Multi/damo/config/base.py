#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import ast
import importlib
import os
import pprint
import sys
from abc import ABCMeta
from os.path import dirname, join
from torch.optim.lr_scheduler import CosineAnnealingLR

from easydict import EasyDict as easydict
from tabulate import tabulate

from .augmentations import test_aug, train_aug
from .paths_catalog import DatasetCatalog

miscs = easydict({
    'print_interval_iters': 50,    # print interval
    'output_dir': './workdirs',    # save dir
    'exp_name': os.path.split(os.path.realpath(__file__))[1].split('.')[0],
    'seed': 1234,                  # rand seed for initialize
    'eval_interval_epochs': 10,    # evaluation interval
    'ckpt_interval_epochs': 10,    # ckpt saving interval
    'num_workers': 4,
})

train = easydict({
    'ema': True,
    'ema_momentum': 0.9998,
    'warmup_start_lr': 0,
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 200,
        'eta_min': 1e-5,
    },
    'min_lr_ratio': 0.05,
    'batch_size': 64,
    'total_epochs': 600,
    'warmup_epochs': 5,
    'no_aug_epochs': 16,
    'resume_path': None,
    'finetune_path': None,
    'augment': train_aug,
    'optimizer': {
        'momentum': 0.9,
        'name': "SGD",
        'weight_decay': 5e-4,
        'nesterov': True,
        'lr': 0.04,
    },
})

test = easydict({
    'augment': test_aug,           # augmentation config for testing
    'batch_size': 128,             # testing batch size
})

dataset = easydict({
    'paths_catalog': join(dirname(__file__), 'paths_catalog.py'),
    'train_ann': ('coco_2017_train', ),
    'val_ann': ('coco_2017_val', ),
    'data_dir': None,
    'aspect_ratio_grouping': False,
    'class_names': None,
})


class Config(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.model = easydict({'backbone': None, 'neck': None, 'head': None})
        self.train = train
        self.test = test
        self.dataset = dataset
        self.miscs = miscs

    def get_data(self, name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=join(data_dir, attrs['img_dir']),
                ann_file=join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format dataset now!')

    def __repr__(self):
        table_header = ['keys', 'values']
        exp_table = [(str(k), pprint.pformat(v, compact=True))
                     for k, v in vars(self).items() if not k.startswith('_')]
        return tabulate(exp_table, headers=table_header, tablefmt='fancy_grid')

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)

    def read_structure(self, path):

        with open(path, 'r') as f:
            structure = f.read()

        return structure


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(
            os.path.basename(config_file).split('.')[0])
        exp = current_config.Config()
    except Exception:
        raise ImportError(
            "{} doesn't contains class named 'Config'".format(config_file))
    return exp


def parse_config(config_file):
    """
    get config object by file.
    Args:
        config_file (str): file path of config.
    """
    assert (config_file is not None), 'plz provide config file'
    if config_file is not None:
        return get_config_by_file(config_file)
