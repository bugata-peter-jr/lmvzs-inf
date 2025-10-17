# -*- coding: utf-8 -*-
"""
Created on Tue May 23 08:35:52 2023

@author: bugatap
"""

import math
import enum

import torch

from typing import List, Sequence

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

    def apply(self, x, d):
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == Initialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            torch.nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == Initialization.NORMAL:
            torch.nn.init.normal_(x, std=d_sqrt_inv)


# special dense layer with reverse forward
class Dense(torch.nn.Linear):

    def forward_rev(self, x:Tensor) -> Tensor:
        if self.bias is not None:
            x -= self.bias
        #print(x.shape, self.weight.shape)
        return torch.matmul(x, self.weight)


# special conversion from long to bits
class Binary(torch.nn.Module):
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


# special conversion from long to bits
class BinaryOld(torch.nn.Module):
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

        if self.level_bits is None or len(self.level_bits) == 0:
            return x.float()

        start = 0
        result = torch.zeros(x.size(0), x.size(1)*2, dtype=torch.long, device=x.device)
        for lb in self.level_bits:
            slice = x[:, start:start+lb]
            result[:, start*2:start*2+lb] = slice
            idxs = (slice.sum(dim=1) == 0)
            # print('slice:', slice)
            # print('idxs:', idxs)
            slice[idxs, :] = True
            # print('slice2:', slice)
            result[:, start*2+lb:start*2+2*lb] = slice.logical_not()
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

        # ucitelny embedding
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        if self.norm_emb:
            x = torch.nn.functional.normalize(x, dim=-1)

        return x


# network s rozsirenym Binary Encoding
class NetworkBE(torch.nn.Module):
    def __init__(self, hsize_sim:int, emb_size:int, level_bits:list[int],
                 n_categories:int, norm_emb:bool = False, initialization:str='none'):

        super().__init__()
        self.level_bits = level_bits
        self.n_classes = len(level_bits) + 1

        self.emb = Embedding(emb_size, level_bits, norm_emb)
        self.activation = torch.nn.ReLU()

        self.sim_hidden = torch.nn.Linear(emb_size, hsize_sim)
        self.sim_head = torch.nn.Linear(hsize_sim, self.n_classes)
        #self.sim_head = torch.nn.Linear(emb_size, self.n_classes)

        self.level_head = torch.nn.Linear(emb_size, self.n_classes)

        self.parent_head = torch.nn.Linear(emb_size, n_categories)

    def freeze_emb(self):
        self.emb.freeze()

    def forward(self, x1, x2):
        x1 = self.emb(x1)
        x2 = self.emb(x2)

        #emb_concat = torch.cat((x1, x2), dim=-1)
        emb_mean = (x1 + x2) / 2

        sim_result = self.sim_hidden(emb_mean)
        sim_result = self.activation(sim_result)
        sim_result = self.sim_head(sim_result)
        #sim_result = self.sim_head(emb_concat)

        level_result1 = self.level_head(x1)

        parent_result1 = self.parent_head(x1)

        return sim_result, level_result1, parent_result1

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


# siet pre vek
# so standardnym kategorialnym embeddingom, ale normalizovanym
class AgeNetwork(torch.nn.Module):
    def __init__(self, n_categories:int, emb_size:int, bias:bool=False, 
           norm_emb:bool = False, initialization:str='none'):

        super().__init__()

        self.emb = CatE(n_categories, emb_size, bias, initialization, norm_emb)

        self.hidden = torch.nn.Linear(emb_size, emb_size)
        self.act = torch.nn.ReLU()

        self.prev_age_out = torch.nn.Linear(emb_size, n_categories)
        self.next_age_out = torch.nn.Linear(emb_size, n_categories)
        self.age_reg_out = torch.nn.Linear(emb_size, 1)
        self.age_diff_out = torch.nn.Linear(emb_size, 1)
        self.age_ph_out = torch.nn.Linear(emb_size, 1)
        self.age_ph_diff_out = torch.nn.Linear(emb_size, 1)
        self.age_range_out = torch.nn.Linear(emb_size, 9)
        self.age_range_diff_out = torch.nn.Linear(emb_size, 1)

    def freeze_emb(self):
        self.emb.freeze()

    def forward(self, x1:Tensor, x2:Tensor) -> Sequence[Tensor]:
        x1 = self.emb(x1)
        x2 = self.emb(x2)

        emb_diff = x1 - x2

        prev_age_out = self.prev_age_out(x1)
        next_age_out = self.next_age_out(x1)
        age_reg_out = self.age_reg_out(x1)
        age_diff_out = self.age_diff_out(emb_diff)
        age_ph_out = self.age_ph_out(x1)
        age_ph_diff_out = self.age_ph_diff_out(emb_diff)
        age_range_out = self.age_range_out(x1)

        #emb_cat = torch.cat([x1, x2], dim=1)

        # hidden a nelinearitu aplikujeme len pri poslednej ulohe
        #emb_cat = self.hidden(emb_cat)
        #emb_cat = self.act(emb_cat)

        emb_diff2 = self.hidden(emb_diff)
        emb_diff2 = self.act(emb_diff2)

        age_range_diff_out = self.age_range_diff_out(emb_diff2)

        return prev_age_out, next_age_out, age_reg_out, age_diff_out, age_ph_out, age_ph_diff_out, age_range_out, age_range_diff_out


