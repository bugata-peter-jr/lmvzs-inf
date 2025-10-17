# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:24:05 2023

@author: bugatap
"""
import os
import sys

import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Any, Union
from numpy.typing import NDArray

from embedding_hierarchie.utils import compute_parent, compute_level_masks

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

Tensor = torch.Tensor
Network = torch.nn.Module
Scheduler = Union[torch.optim.lr_scheduler.LRScheduler, None]
Optimizer = torch.optim.Optimizer
StateDict = Dict[str, Any]
Device = torch.device


def compute_pe(bin_code: int, level: int, level_masks: List[int], bin_codes: NDArray[np.int64]) -> List[int]:
    pe: List[int] = []

    # vlozim label daneho binarneho kodu
    idxs = np.argwhere(bin_codes == bin_code).flatten()
    pe.append(idxs[0])

    # rekurentne vkladam kody rodicov
    b_code = bin_code
    while True:
        parent_bin_code = compute_parent(b_code, level, level_masks)
        if parent_bin_code == 0:
            pe.append(0)
            break
        parent_idxs = np.argwhere(bin_codes == parent_bin_code).flatten()
        pe.append(parent_idxs[0])
        b_code = parent_bin_code
        level -= 1

    # otocim postupnost
    pe = pe[::-1]

    # doplnim nuly za chybajuce urovne
    while len(pe) <= len(level_masks):
        pe.append(0)

    return pe


def add_pe(df_cs: pd.DataFrame, level_bits: list[int]):
    df_cs.sort_values(by='kod_hier', inplace=True)

    levels = np.array(df_cs.uroven.values, dtype=np.int8)
    bin_codes = np.array(df_cs.kod_hier.values, dtype=np.int64)
    level_masks = compute_level_masks(level_bits)

    parents = np.zeros(len(bin_codes), dtype=np.int64)
    pe: List[List[int]] = []
    for i in range(len(bin_codes)):
        parent_bin_code = compute_parent(bin_codes[i], levels[i], level_masks)
        parent_idxs = np.argwhere(bin_codes == parent_bin_code).flatten()
        parents[i] = parent_idxs[0]
        item_pe = compute_pe(bin_codes[i], levels[i], level_masks, bin_codes)
        if len(item_pe) < 3:
            print('PE:', item_pe)
        pe.append(item_pe)

    df_cs['pe'] = pe


# porovna hierarchicke binarne kody (len po uroven target-u)
def compare_hierarchy_codes(pred: Tensor, target: Tensor, level_bits: List[int], strict: bool = False):
    if strict:
        return target.eq(pred)

    target = target.reshape(-1, 1)
    pred = pred.reshape(-1, 1)

    masks: List[Tensor] = []
    zero_mask: List[Tensor] = []
    code_len = sum(level_bits)
    ones = 0
    shift = code_len

    for lb in level_bits:
        mask = int('0b' + lb * '1', 2)
        data = torch.tensor(mask, dtype=torch.long, device=pred.device)
        ones += lb
        zeros = code_len - ones
        zmask = int('0b' + ones * '1' + zeros * '0', 2)
        zdata = torch.tensor(zmask, dtype=torch.long, device=pred.device)
        shift -= lb
        data = data.bitwise_left_shift(shift)
        masks.append(data)
        zero_mask.append(zdata)

    # masks_t = torch.LongTensor(masks, device=pred.device).reshape(1, -1)
    masks_t = torch.tensor(masks, device=pred.device, dtype=torch.long).reshape(1, -1)
    target_levels = target.bitwise_and(masks_t)
    target_levels = target_levels.ne(0).sum(-1)
    # print('target_levels:', target_levels)

    for i in range(len(level_bits)-1):
        # print(i, bin(zero_mask[i].item()))
        idx = target_levels.eq(i+1)
        # print(bin(pred[idx].item()))
        pred[idx] = pred[idx].bitwise_and(zero_mask[i])
        # print(bin(pred[idx].item()))

    result = target.eq(pred)
    # ak je target 0, tak lubovolnu predikciu povazujeme za spravnu
    result[target == 0] = True
    return result.flatten()

class Statistics:
    def __init__(self, labels: List[np.int32], counts: List[np.int64], device: torch.device):
        self.labels = torch.Tensor(labels).to(dtype=torch.int64, device=device)
        cnts = torch.Tensor(counts).to(dtype=torch.int64, device=device)
        self.probs = cnts / torch.sum(cnts)
        return


class StatisticsExt(Statistics):
    def __init__(self, labels: List[np.int32], counts: List[np.int64],
                 bin_codes: List[np.int64], pe: List[List[int]], device: torch.device):
        super().__init__(labels, counts, device=device)
        self.bin_codes = torch.Tensor(bin_codes).to(dtype=torch.int64, device=device)
        self.pe = torch.Tensor(pe).to(dtype=torch.int64, device=device)
        return


def compute_stats(df: pd.DataFrame, cat_vars: List[str], device: torch.device) -> Dict[str, Statistics]:
    var_dict: Dict[str, Statistics] = {}

    for var in cat_vars:
        labels: List[np.int32] = []
        counts: List[np.int64] = []
        bin_codes: List[np.int64] = []
        pe: List[List[int]] = []

        for val, df_part in df.groupby(var):
            count = np.int64(df_part.pocet.sum())
            labels.append(np.int32(val))  # type: ignore
            counts.append(count)
            if var.endswith('_l'):
                var_name = var[3:-2]  # vynechame "id_" zo zaciatku a "_l" z konca
                val_b = df_part['id_' + var_name + '_b'].iat[0]
                pe_val = df_part['pe_' + var_name].iat[0]
                bin_codes.append(np.int64(val_b))
                pe.append(pe_val)

        if not var.endswith('_l'):
            var_dict[var] = Statistics(labels, counts, device)
        else:
            var_dict[var] = StatisticsExt(labels, counts, bin_codes, pe, device)

    return var_dict


if __name__ == '__main__':

    from .config import Config

    # instantiate config
    cfg = Config()

    #target = torch.LongTensor([1066385, 1066384, 1066240, 1064960, 1048576, 1066240, 1064960, 1066384])
    #pred = torch.LongTensor([1066385, 1066385, 1066385, 1066385, 1066385, 1066240, 1066240, 1066240])

    target = torch.LongTensor([0, 0, 96, 97, 97])
    pred = torch.LongTensor([97, 0, 97, 96, 97])

    result = compare_hierarchy_codes(pred, target, cfg.level_bits['id_odb_l'])
    print(result.shape, result)
