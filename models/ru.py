#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /RelatingUp/models/ru.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2024-01-25 17:43:09
-------------------------------------------------
Change Activity :
  LastEditTime  : 2024-01-25 17:43:10
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from models.base import LModule


class RU(LModule):
    def __init__(self, net, 
                 num_classes: int, 
                 lr=0.001,
                 weight_decay=5e-4,
                 alpha: float = 0.001,
                 beta: float = 1e-5,
                 temperature: float = 3,
                 **kwargs) -> None:
        super().__init__(net=net,   
                         num_classes=num_classes,
                         lr=lr, 
                         weight_decay=weight_decay, 
                         kwargs=kwargs)

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
    def train_criterion(self, logits_o, logits_e, embed_o, embd_e, targets):
        ce_o = self.criterion(logits_o, targets)
        ce_e = self.criterion(logits_e, targets)
        kl = F.kl_div(input=F.log_softmax(logits_o, -1),
              target=F.softmax(logits_e.detach() / self.temperature, -1), reduction="batchmean") * (self.temperature ** 2)
        feature_loss = self.feature_loss_function(embed_o, embd_e.detach())
        
        self.log("train/ce_o_loss", ce_o.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=logits_e.shape[0])
        self.log("train/ce_e_loss", ce_e.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=logits_e.shape[0])
        self.log("train/kl_loss", kl.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=logits_e.shape[0])
        self.log("train/feature_loss", feature_loss.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=logits_e.shape[0])

        return (1 - self.alpha) * (ce_o + ce_e) + self.alpha * kl + self.beta * feature_loss
    
    @staticmethod
    def feature_loss_function(fea, target_fea):
        loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).sum()
    
    def model_step(self, batch: Batch):
        if self.training:
            logits_o, logits_e, embd_o, embd_e = self.net(batch)
            loss = self.train_criterion(logits_o, logits_e, embd_o, embd_e, batch.y)
            preds = torch.argmax(logits_o, dim=-1)
        else:
            logits = self.net(batch)
            loss = self.criterion(logits, batch.y)
            preds = torch.argmax(logits, dim=-1)
        
        return loss, preds, batch.y
