#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /RelatingUp/main.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2024-01-25 16:35:33
-------------------------------------------------
Change Activity :
  LastEditTime  : 2024-01-25 16:35:33
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""

import argparse
import json
import os
from email.policy import default
from functools import reduce
from typing import List

import lightning as L
import numpy as np
from prettytable import PrettyTable

from dataset.tudataset import TUDatasetModule
from models.backbone import Backbone
from models.gcn import GCN, GCNRU
from models.gin import GIN, GINRU
from models.ru import RU


def reduce_metric_dict(metric_dict: List, reduce_='mean') -> dict:
    """Reduce the metric dictionary by the given `reduce` method.
    """
    reduced_dict = reduce(lambda acc, curr: {**acc, **{k: acc.get(k, []) + [v.item() if hasattr(v, "item") else v] for k, v in curr.items()}},
                          metric_dict,
                          {})
    if reduce_ == 'mean':
        return {k: np.mean(v) for k, v in reduced_dict.items()}
    elif reduce_ == 'sum':
        return {k: np.sum(v) for k, v in reduced_dict.items()}
    elif reduce_ == 'std':
        return {k: np.std(v) for k, v in reduced_dict.items()}
    
    
def pretty_table(mean_metrics: dict, std_metrics: dict, metrics_head=['loss', 'acc']) -> str:
    metrics_head = metrics_head
    table = PrettyTable(['Phase'] + metrics_head)

    for phase in ['train', 'val', 'test']:
        row = [phase]
        for m in metrics_head:
            if m == "loss":
                frac = 1
            else:
                frac = 100
            avg_ = '{:0>5.2f}'.format(mean_metrics[f'{phase}/{m}'] * frac)
            std_ = '{:0>5.2f}'.format(std_metrics[f'{phase}/{m}'] * frac)
            row.append(avg_ + '/' + std_)
        table.add_row(row)

    return table.get_string()


def train_fold(fold, args):
    L.seed_everything(args.seed, workers=True)
    datamodule = TUDatasetModule(data_dir = root,
                            name="MUTAG",
                            batch_size=args.batch_size,
                            num_workers=3, 
                            pin_memory=True,
                            n_splits=args.n_splits,
                            n_repeats=args.n_repeats,
                            persistent_workers=True,
                            random_state=args.seed)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.setup_fold_index(fold_index=fold)
    
    if args.model == "GCN":
        net = GCN(input_dim=datamodule.dim_features,
                  hidden_dim=args.hidden_dim,
                  out_dim=datamodule.dim_targets, 
                  num_layers=args.num_layers, pooling="add")
    elif args.model == "GIN":
        net = GIN(input_dim=datamodule.dim_features,
                  hidden_dim=args.hidden_dim,
                  out_dim=datamodule.dim_targets, 
                  num_layers=args.num_layers, pooling="add")
    elif args.model == "GCNRU":
        net = GCNRU(input_dim=datamodule.dim_features,
            hidden_dim=args.hidden_dim,
            out_dim=datamodule.dim_targets, 
            num_layers=args.num_layers, nhead=4, pooling="add")
    elif args.model == "GINRU":
        net = GINRU(input_dim=datamodule.dim_features,
            hidden_dim=args.hidden_dim,
            out_dim=datamodule.dim_targets, 
            num_layers=args.num_layers, nhead=4, pooling="add")
        
    if args.model.endswith("RU"):
        model = RU(net=net,
                   lr=args.lr,
                   weight_decay=args.weight_decay,
                   num_classes=datamodule.dim_targets,
                   alpha=args.alpha,
                   beta=args.beta,
                   temperature=args.temp)
    else:
        model = Backbone(net=net,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        num_classes=datamodule.dim_targets)
    
    callbacks = [
            L.pytorch.callbacks.EarlyStopping(monitor="val/acc", 
                                              patience=args.patience,
                                              mode="max"),
            L.pytorch.callbacks.ModelCheckpoint(monitor="val/acc", save_top_k=1, mode="max"),
        ]
    trainer = L.pytorch.trainer.Trainer(default_root_dir=os.path.join(output_dir, f"FOLD_{fold}"), 
                                        min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                        accelerator="gpu" if args.cuda else "cpu",
                                        devices=1,
                                        check_val_every_n_epoch=1,
                                        log_every_n_steps=1,
                                        num_sanity_val_steps=0,
                                        deterministic=True,
                                        benchmark=True,
                                        gradient_clip_val=args.gradient_clip_val,
                                        callbacks=callbacks
                                        )
    
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    test_metrics = trainer.callback_metrics
    
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict
    

def assessment(args):
    metrics = []
    
    for fold in range(args.n_repeats * args.n_splits):
        metric = train_fold(fold=fold, args=args)
        metrics.append(metric)
    
    mean_metrics = reduce_metric_dict(metric_dict=metrics, reduce_="mean")
    std_metrics = reduce_metric_dict(metric_dict=metrics, reduce_="std")
    return mean_metrics, std_metrics

    
def main(args):
    
    if not os.path.exists(os.path.join(output_dir, "result.json")):
        mean_metrics, std_metrics = assessment(args=args)
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump({"mean": mean_metrics, "std": std_metrics}, f)
    else:
        with open(os.path.join(output_dir, "result.json"), "r") as f:
            metrics = json.load(f)
            mean_metrics, std_metrics = metrics["mean"], metrics["std"]

    table = pretty_table(mean_metrics=mean_metrics, std_metrics=std_metrics)
    print(table)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Relating-Up')
    parser.add_argument("--dataset", type=str, required=True, help='Name of dataset')
    parser.add_argument("--model", type=str, choices=["GCN", "GIN", "GCNRU", "GINRU"], required=True, help="Name of model")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed (default: 2023)")
    
    parser.add_argument("--n_splits", type=int, default=10, help="Number of splits")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of times cross-validation needs to be repeated")
    
    parser.add_argument("--batch_size", type=int, default=128, help="Input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 penalty) (default: 5e-4)")
    parser.add_argument("--gradient_clip_val", type=float, default=1, help="The value at which to clip gradients")
    parser.add_argument("--patience", type=int, default=100, help="Number of validation epochs with no improvement after which training will be stopped")
    parser.add_argument("--min_epochs", type=int, default=150, help="Force training for at least `min_epochs` epochs")
    parser.add_argument("--max_epochs", type=int, default=300, help="Stop training once `max_epochs` is reached")
    
    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of hidden units (default: 128)")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.1, help="The parameter controls the balance between the Cross Entropy loss and the distillation loss")
    parser.add_argument("--beta", type=float, default=1e-6, help="Weight of representation hints loss")
    parser.add_argument("--temp", type=float, default=3, help="Temperature to smooth the logits")
    
    parser.add_argument("--cuda", action="store_true", help="")
    
    args = parser.parse_args()
    
    root = os.path.join("data", "TUDataset")
    output_dir = os.path.join("output", args.model, args.dataset,
                              f"lr={args.lr},weight_decay={args.weight_decay},hidden_dim={args.hidden_dim},num_layers={args.num_layers},alpha={args.alpha},beta={args.beta},temp={args.temp},gradient_clip_val={args.gradient_clip_val}")
    main(args)