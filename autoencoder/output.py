# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:07:06 2023

@author: bugatap
"""

import numpy as np
import argparse

import torch
from pytorch_model_summary import summary  # type: ignore

from typing import Tuple, Callable, Dict
Tensor = torch.Tensor


class OutputBlock(torch.nn.Module):
    def __init__(self, hsize: int, n_cat: int, emb_size: int, 
                 activation: Callable[[], torch.nn.Module] = torch.nn.ReLU):
        super().__init__()

        self.proj = torch.nn.Linear(hsize, emb_size)
        self.activation = activation()
        self.norm = torch.nn.LayerNorm(emb_size)
        self.output = torch.nn.Linear(emb_size, n_cat)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.output(x)
        return x


# vystup pre riadok
class RowOutput(torch.nn.Module):
    def __init__(self, hsize: int, n_cats: Dict[str, int], emb_size: Dict[str, int], has_proj: bool=True, 
                 hier_age: bool=False, activation: Callable[[], torch.nn.Module] = torch.nn.ReLU):
        super().__init__()

        if has_proj:
            self.norm = None
            self.dg_output = OutputBlock(hsize, n_cats['id_diag_l'], emb_size['id_diag_d'], activation)
            self.prod_output = OutputBlock(hsize, n_cats['id_prod_l'], emb_size['id_prod_d'], activation)
            self.odb_output = OutputBlock(hsize, n_cats['id_odb_l'], emb_size['id_odb_d'], activation)
            self.age_output = OutputBlock(hsize, n_cats['vek'], emb_size['vek_d'] * 2, activation)
            self.gender_output = OutputBlock(hsize, n_cats['pohlavie'], emb_size['pohlavie'], activation)
        else:
            self.norm = torch.nn.LayerNorm(hsize)
            self.dg_output = torch.nn.Linear(hsize, n_cats['id_diag_l'])
            self.prod_output = torch.nn.Linear(hsize, n_cats['id_prod_l'])
            self.odb_output = torch.nn.Linear(hsize, n_cats['id_odb_l'])
            self.age_output = torch.nn.Linear(hsize, n_cats['vek'])
            self.gender_output = torch.nn.Linear(hsize, n_cats['pohlavie'])

        self.hier_age = hier_age

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.norm is not None:
            x = self.norm(x)        
        
        x_dg = self.dg_output(x)
        x_dg = torch.softmax(x_dg, dim=-1)

        x_prod = self.prod_output(x)
        x_prod = torch.softmax(x_prod, dim=-1)

        x_odb = self.odb_output(x)
        x_odb = torch.softmax(x_odb, dim=-1)

        x_age = self.age_output(x)
        if self.hier_age:
            x_age = torch.softmax(x_age, dim=-1)   

        x_gender = self.gender_output(x)

        return x_dg, x_prod, x_odb, x_age, x_gender
    