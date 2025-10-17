# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:07:06 2023

@author: bugatap
"""

import os
import sys

import numpy as np
import argparse

import math
import enum

import torch

from embedding_hierarchie.network import Binary

from typing import List, Dict
Tensor = torch.Tensor

# enum class for initialization
class Initialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    NONE = 'none'

    @classmethod
    def from_str(cls, initialization: str) -> 'Initialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in Initialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int):
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == Initialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            torch.nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == Initialization.NORMAL:
            torch.nn.init.normal_(x, std=d_sqrt_inv)


# standardny emdedding kategorialnej premennej s bias
class CatE(torch.nn.Module):
    def __init__(self, n_cats: int, emb_size: int, bias: bool = False, initialization: str = 'none', norm:bool=False) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.emb = torch.nn.Embedding(n_cats, emb_size)
        self.bias = torch.nn.Parameter(torch.zeros(emb_size)) if bias else None

        self.norm = norm

        # init weights
        initialization_ = Initialization.from_str(initialization)
        for parameter in [self.emb.weight, self.bias]:
            if parameter is not None:
                assert isinstance(parameter, Tensor)
                initialization_.apply(parameter, emb_size)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        if self.bias is not None:
            x += self.bias
        if self.norm:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x


class PathEncoding(torch.nn.Module):
    def __init__(self, n_cats: int):
        super().__init__()
        self.n_cats = n_cats

    def forward(self, x: Tensor) -> Tensor:
        #print('x.shape:', x.shape)
        bs = x.size(0)

        #shape = (bs, self.n_cats)
        shape = list(x.shape)
        shape[-1] = self.n_cats

        result = torch.zeros(size=shape, device=x.device, dtype=x.dtype)

        # inplace scatter
        #result.scatter_(dim=1, index=x, src=torch.ones_like(x))
        result.scatter_(dim=-1, index=x, src=torch.ones_like(x))

        return result.float()


class PathEmbedding(torch.nn.Module):
    def __init__(self, n_cats: int, emb_size: int):
        super().__init__()
        self.pe = PathEncoding(n_cats)
        self.dense = torch.nn.Linear(n_cats, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pe(x)
        x = self.dense(x)
        return x


# special conversion from long to bits
class BinaryCopy(torch.nn.Module):
    def __init__(self, bits: int, level_bits: List[int]):
        super().__init__()
        self.level_bits = level_bits
        self.mask = torch.nn.parameter.Parameter(data=2**torch.arange(bits, dtype=torch.long), requires_grad=False)
        assert level_bits is None or bits == sum(level_bits), 'Level bits are not correct'

    def forward(self, x: Tensor) -> Tensor:
        # print("x.shape:", x.shape)
        x = x.unsqueeze(-1).bitwise_and(self.mask).ne(0)
        x = x.flip(dims=[-1])
        # print("x.binary.shape:", x.shape)
        # print('x.binary:', x.to(dtype=torch.int))

        if self.level_bits is None or len(self.level_bits) == 0:
            return x.float()


        shape = list(x.shape)
        shape[-1] = 2 * shape[-1] # zdojnasobime poslednu dimenziu
        result = torch.zeros(shape, dtype=torch.long, device=x.device)

        start = 0
        for lb in self.level_bits:
            # slice = x[:, start:start+lb]
            slice_idx = torch.arange(start, start+lb, device=x.device)
            slice = x.index_select(dim=-1, index=slice_idx)
            # print('slice:', slice)

            # result[:, start*2:start*2+lb] = slice
            dst_idx = torch.arange(start*2, start*2+lb, device=x.device)
            result.index_add_(dim=-1, index=dst_idx, source=slice.to(dtype=torch.long))

            idxs = (slice.sum(dim=-1) == 0)
            # print('slice:', slice)
            # print('idxs:', idxs)
            slice[idxs, :] = True
            # print('slice2:', slice)

            # result[:, start*2+lb:start*2+2*lb] = slice.logical_not()
            dst_idx = torch.arange(start*2+lb, start*2+2*lb, device=x.device)
            result.index_add_(dim=-1, index=dst_idx, source=slice.logical_not().to(dtype=torch.long))

            start += lb

        return result.float()


# modul pre hladanie embedingu na zaklade binarneho encodingu
class Embedding(torch.nn.Module):
    def __init__(self, emb_size:int, level_bits:list[int], norm_emb: bool):

        super().__init__()
        self.level_bits = level_bits
        inp_size = sum(level_bits)

        self.norm_emb = norm_emb

        self.binary = Binary(inp_size, self.level_bits)

        self.linear1 = torch.nn.Linear(inp_size*2, inp_size*8, bias=True)
        self.activation = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(inp_size*8, emb_size, bias=True)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # prevod na binarny tvar
        x = self.binary(x)
        #print('binary:', x)

        # ucitelny embedding
        x = self.linear1(x)
        #print('linear1:', x)
        x = self.activation(x)
        #print('act:', x)
        x = self.linear2(x)
        #print('linear2:', x)

        if self.norm_emb:
            #x = torch.nn.functional.normalize(x)
            x = torch.nn.functional.normalize(x, dim=-1)
            #print('norm:', torch.abs(x).sum(dim=-1))

        return x


# embedding riadku
class RowEmbedding(torch.nn.Module):
    def __init__(self, n_cats: Dict[str, int], emb_size: Dict[str, int], level_bits: Dict[str, List[int]],
                 row_emb_size: int, bias: bool = False, norm_emb: bool = False, initialization: str = 'none'):
        super().__init__()

        self.output_size = 0
        self.row_emb_size = row_emb_size

        emb_size_dg_t, emb_size_dg_d = emb_size['id_diag_t'], emb_size['id_diag_d']
        self.dg_emb_t = Embedding(emb_size=emb_size_dg_t, level_bits=level_bits['id_diag_l'], norm_emb=norm_emb)
        self.dg_emb_d = PathEmbedding(n_cats=n_cats['id_diag_l'], emb_size=emb_size_dg_d)
        self.output_size += emb_size_dg_t + emb_size_dg_d

        emb_size_prod_t, emb_size_prod_d = emb_size['id_prod_t'], emb_size['id_prod_d']
        self.prod_emb_t = Embedding(emb_size=emb_size_prod_t, level_bits=level_bits['id_prod_l'], norm_emb=norm_emb)
        self.prod_emb_d = PathEmbedding(n_cats=n_cats['id_prod_l'], emb_size=emb_size_prod_d)
        self.output_size += emb_size_prod_t + emb_size_prod_d

        emb_size_odb_t, emb_size_odb_d = emb_size['id_odb_t'], emb_size['id_odb_d']
        self.odb_emb_t = Embedding(emb_size=emb_size_odb_t, level_bits=level_bits['id_odb_l'], norm_emb=norm_emb)
        self.odb_emb_d = PathEmbedding(n_cats=n_cats['id_odb_l'], emb_size=emb_size_odb_d)
        self.output_size += emb_size_odb_t + emb_size_odb_d

        emb_size_age_t, emb_size_age_d = emb_size['vek_t'], emb_size['vek_d']
        self.age_emb_d = CatE(n_cats=n_cats['vek'], emb_size=emb_size_age_d, bias=bias, initialization=initialization, norm=norm_emb)
        self.age_emb_t = CatE(n_cats=n_cats['vek'], emb_size=emb_size_age_t, bias=bias, initialization=initialization, norm=norm_emb)
        self.output_size += emb_size_age_t + emb_size_age_d

        emb_size_gender = emb_size['pohlavie']
        self.gender_emb = CatE(n_cats=n_cats['pohlavie'], emb_size=emb_size_gender, bias=bias,
                               initialization=initialization, norm=norm_emb)
        self.output_size += emb_size_gender

        self.linear = torch.nn.Linear(self.output_size, self.row_emb_size, bias=bias)  # TODO ci do rovnakeho rozmeru

    def freeze(self):
        self.dg_emb_t.freeze()
        self.prod_emb_t.freeze()
        self.odb_emb_t.freeze()
        self.age_emb_t.freeze()

    def forward(self, x_dg_bin: Tensor, x_prod_bin: Tensor, x_odb_bin: Tensor,
                x_dg_pe: Tensor, x_prod_pe: Tensor, x_odb_pe: Tensor,
                x_age: Tensor, x_gender: Tensor) -> Tensor:
        x_dg1 = self.dg_emb_t(x_dg_bin)
        x_dg2 = self.dg_emb_d(x_dg_pe)
        #print('dg_bin.shape:', x_dg1.shape)
        #print('dg_bin:', x_dg1)
        #print('dg_pe.shape:', x_dg2.shape)
        #print('dg_pe:', x_dg2)

        x_prod1 = self.prod_emb_t(x_prod_bin)
        x_prod2 = self.prod_emb_d(x_prod_pe)

        x_odb1 = self.odb_emb_t(x_odb_bin)
        x_odb2 = self.odb_emb_d(x_odb_pe)

        x_age1 = self.age_emb_t(x_age)
        x_age2 = self.age_emb_d(x_age)

        x_gender = self.gender_emb(x_gender)

        x_final = torch.cat([x_dg1, x_dg2, x_prod1, x_prod2, x_odb1, x_odb2, x_age1, x_age2, x_gender], dim=-1)
        x_final = self.linear(x_final)
        return x_final


if __name__ == '__main__':
    # cesta k suborom
    base_path = '/home/azureuser/localfiles/v2'

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to data folder", default=base_path + '/data')
    parser.add_argument("--models_path", type=str, help="path to models folder", default=base_path + '/models')
    # parser.add_argument("--start_date", type=str, help="start date", default=None)
    parser.add_argument("--remove_flag", type=bool, help="remove flag", default=False)
    args = parser.parse_args()

    data_path = args.data_path
    models_path = args.models_path

    from config import Config

    cfg = Config()

    '''
    level_bits = cfg.level_bits['id_prod_l']
    binary = Binary(sum(level_bits), level_bits)

    # 0110000000010000000100000000100001000000001
    x = torch.LongTensor([3300690772481, 3300690772481])
    print('shape:', x.shape)
    print(binary(x))

    x = torch.LongTensor([[3300690772481, 3300690772481],
                          [3300690772481, 3300690772481]])
    print('shape:', x.shape)
    print(binary(x))
    '''

    device = 'cpu'

    n_cats = {'id_diag_l': 13585, 'id_prod_l': 8384, 'id_odb_l': 256, 'vek': 98, 'pohlavie': 3}

    row_embedding = RowEmbedding(n_cats=n_cats, emb_size=cfg.emb_size, level_bits=cfg.level_bits,
                 row_emb_size=cfg.row_emb_size, bias=cfg.bias, norm_emb=cfg.norm_emb, initialization=cfg.initialization)
    print('RowEmbedding created')

    emb_path = models_path + cfg.emb_dir + '/'
    row_embedding.dg_emb_t.load_state_dict(torch.load(emb_path+cfg.dg_file, map_location=device))
    row_embedding.prod_emb_t.load_state_dict(torch.load(emb_path+cfg.prod_file, map_location=device))
    row_embedding.odb_emb_t.load_state_dict(torch.load(emb_path+cfg.odb_file, map_location=device))
    row_embedding.age_emb_t.load_state_dict(torch.load(emb_path+cfg.age_file, map_location=device))
    print('RowEmbedding loaded')
