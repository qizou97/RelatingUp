#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /RelatingUp/models/backbone.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2024-01-25 16:31:51
-------------------------------------------------
Change Activity :
  LastEditTime  : 2024-01-25 16:31:51
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import torch
from torch.distributed.pipeline.sync.microbatch import Batch

from models.base import LModule


class Backbone(LModule):
    def __init__(self, net, 
                 num_classes: int, 
                 lr=0.001,
                 weight_decay=5e-4,
                 **kwargs) -> None:
        super().__init__(net=net, 
                         num_classes=num_classes,
                         lr=lr, 
                         weight_decay=weight_decay, 
                         kwargs=kwargs)

    def model_step(self, batch: Batch):
        labels = batch.y
        logits = self.net(batch)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels