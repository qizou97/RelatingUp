#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /RelatingUp/models/gcn.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2024-01-25 16:43:46
-------------------------------------------------
Change Activity :
  LastEditTime  : 2024-01-25 16:43:46
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.nn.models import MLP


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, pooling="add") -> None:
        super().__init__()
            
        self.conv = gnn.models.GCN(in_channels=input_dim,
                                   hidden_channels=hidden_dim,
                                   num_layers=num_layers,
                                   dropout=0.5, 
                                   jk="cat")
        if pooling == "add":
            self.pooling = gnn.global_add_pool
        elif pooling == "mean":
            self.pooling = gnn.global_mean_pool
        elif pooling == "max":
            self.pooling = gnn.global_max_pool
        elif pooling == "attention":
            self.pooling = gnn.aggr.AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid()))
            
        self.linear = MLP([hidden_dim, hidden_dim, out_dim], dropout=0.5)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.linear.reset_parameters()
        
    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x=x, edge_index=edge_index, batch=batch)
        emb_o = self.pooling(x, batch)
        return emb_o
    
    def get_prediction(self, emb):
        return self.linear(emb)

    def forward(self, data):
        emb = self.get_embedding(data)
        return self.get_prediction(emb)
    
    
class GCNRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim,
                 num_layers,
                 nhead,
                 dropout = 0.5,
                 pooling="add") -> None:
        super().__init__()
        
        self.conv = gnn.models.GCN(in_channels=input_dim,
                                   hidden_channels=hidden_dim,
                                   num_layers=num_layers,
                                   dropout=0.5, 
                                   jk="cat")
        if pooling == "add":
            self.pooling = gnn.global_add_pool
        elif pooling == "mean":
            self.pooling = gnn.global_mean_pool
        elif pooling == "max":
            self.pooling = gnn.global_max_pool
        elif pooling == "attention":
            self.pooling =gnn.aggr.AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid()))
            
        self.ru = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True), num_layers=1)    
         
        self.fc_o = MLP([hidden_dim, hidden_dim, out_dim], dropout=0.5)
        self.fc_e = MLP([hidden_dim, hidden_dim, out_dim], dropout=0.5)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.fc_o.reset_parameters()
        self.fc_e.reset_parameters()
        
        for layer in self.ru.modules():
            if hasattr(layer, "_reset_parameters"):
                layer._reset_parameters()
            elif hasattr(layer, "reset_parameter"):
                layer.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x=x, edge_index=edge_index, batch=batch)
        emb_o = self.pooling(x, batch)
        pred_o = self.fc_o(emb_o)
        if not self.training:
            return pred_o            
        else:
            emb_e = self.ru(emb_o.unsqueeze(dim=0)).squeeze(dim=0)
            pred_e = self.fc_e(emb_e)
            return pred_o, pred_e, emb_o, emb_e

    @torch.no_grad()
    def get_attention(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x=x, edge_index=edge_index, batch=batch)
        emb_o = self.pooling(x, batch)
        x = emb_o.unsqueeze(0)
        attns = self.ru.layers[0].self_attn(x, x, x, need_weights=True)[1]
        return attns
