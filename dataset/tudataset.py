#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /RelatingUp/dataset/tudataset.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2024-01-25 16:18:03
-------------------------------------------------
Change Activity :
  LastEditTime  : 2024-01-25 16:27:47
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import json
import os
from abc import abstractmethod
from typing import Callable, List, Optional

import lightning as L
import numpy as np
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class GraphLevelDatasetModule(L.LightningModule):
    def __init__(self,
                 data_dir: str,
                 name: str,
                 batch_size: int = 128,
                 holdout_size: float = 0.1,
                 n_splits: int = None,
                 n_repeats: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 pre_transforms: Optional[Callable] = None,
                 random_state: int = 42, 
                 drop_last: bool = False):
        super(GraphLevelDatasetModule, self).__init__()
        
        self.data_dir: str = data_dir
        self.name: str = name
        self.batch_size: int = batch_size
        self.holdout_size: float = holdout_size
        self.n_splits: int = n_splits
        self.n_repeats: int = n_repeats or 1
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers = persistent_workers
        self.pre_transform: Optional[Callable] = pre_transforms
        self.random_state: Optional[int] = random_state

        self.dataset = None
        self.drop_last = drop_last

        self.SPLIT_FILENAME = None
        self.dataset: Optional[Dataset] = None
        self.splits: List[dict] = []

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def dim_targets(self):
        return self.dataset.num_classes

    @property
    def dim_features(self):
        return self.dataset.num_features

    @property
    def num_graphs(self):
        return len(self.dataset)

    @property
    def targets(self):
        return self.dataset.data.y

    @abstractmethod
    def setup_fold_index(self, fold_index):
        pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, drop_last=self.drop_last)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, drop_last=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, drop_last=False)


class TUDatasetModule(GraphLevelDatasetModule):
    def __init__(self,
                 data_dir: str,
                 name: str,
                 batch_size: int = 128,
                 holdout_size: float = 0.1,
                 n_splits: int = None,
                 n_repeats: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 pre_transforms: Optional[Callable] = None,
                 random_state: int = 2023,
                 drop_last: bool = False):
        super(TUDatasetModule, self).__init__(data_dir=data_dir,
                                              name=name,
                                              batch_size=batch_size,
                                              holdout_size=holdout_size,
                                              n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              persistent_workers=persistent_workers,
                                              pre_transforms=pre_transforms,
                                              random_state=random_state, drop_last=drop_last)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        TUDataset(root=self.data_dir, name=self.name, pre_transform=self.pre_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.name.startswith("IMDB"):
            transforms =[OneHotDegree(in_degree=True, cat=True, max_degree=600), T.AddSelfLoops()]
        elif self.name.startswith("REDDIT"):
            transforms = [T.Constant(), T.AddSelfLoops()]
        else:
            transforms =[T.AddSelfLoops()]
        
        self.dataset = TUDataset(root=self.data_dir, name=self.name, 
                                 pre_transform=self.pre_transform, transform=T.Compose(transforms))
        self.setup_splits()

    def setup_splits(self) -> None:
        SPLIT_FILE = os.path.join(self.data_dir, self.name, 'splits')
        if not os.path.exists(SPLIT_FILE):
            os.makedirs(SPLIT_FILE)
        self.SPLIT_FILENAME = os.path.join(SPLIT_FILE,
                                           f'splits-{self.n_splits}_'
                                           f'repeats-{self.n_repeats}_'
                                           f'seed-{self.random_state}.json')

        if not os.path.exists(self.SPLIT_FILENAME):
            
            X = np.arange(self.num_graphs)
            Y = self.targets.numpy().reshape(-1, )

            if self.n_splits is None:  # holdout split
                assert self.holdout_size is not None
                assert self.holdout_size > 0
                for _ in range(self.n_repeats):
                    train_o_split, test_split = train_test_split(X,
                                                                 stratify=Y,
                                                                 test_size=self.holdout_size,
                                                                 random_state=self.random_state + _,
                                                                 shuffle=True)
                    train_split, val_split = train_test_split(train_o_split,
                                                              stratify=Y[train_o_split],
                                                              test_size=self.holdout_size,
                                                              random_state=self.random_state + _,
                                                              shuffle=True)
                    split = {'test': test_split.tolist(), 
                             'train': train_split.tolist(), 
                             'val': val_split.tolist()}
                    self.splits.append(split)

            else:  # cross-validation splits
                for _ in range(self.n_repeats):
                    kfold = StratifiedKFold(n_splits=self.n_splits,
                                            shuffle=True,
                                            random_state=self.random_state + _)
                    for train_o_split, test_split in kfold.split(X, y=Y):
                        np.random.shuffle(train_o_split)
                        np.random.shuffle(test_split)
                        
                        train_split, val_split = train_test_split(train_o_split, 
                                                                  stratify=Y[train_o_split],
                                                                test_size=self.holdout_size,
                                                                random_state=self.random_state + _,
                                                                shuffle=True)
                        
                        split = {'test': test_split.tolist(), 
                                 'train': train_split.tolist(), 
                                 'val': val_split.tolist()}

                        self.splits.append(split)

            with open(self.SPLIT_FILENAME, 'w') as f:
                json.dump(self.splits, f, cls=NumpyEncoder)
        else:
            with open(self.SPLIT_FILENAME, 'r') as f:
                self.splits = json.load(f)

    def setup_fold_index(self, fold_index: int) -> None:

        train_indices, val_indices, test_indices = self.splits[fold_index]['train'], \
                                                self.splits[fold_index]['val'], \
                                                self.splits[fold_index]['test']
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset[train_indices], \
                                                                self.dataset[val_indices], \
                                                                self.dataset[test_indices]
