# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:24:05 2023

@author: bugatap
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from typing import List, Dict, Union, Any, Tuple
from numpy.typing import ArrayLike, NDArray

Tensor = torch.Tensor
Network = torch.nn.Module
Scheduler = Union[torch.optim.lr_scheduler.LRScheduler, None]
Optimizer = torch.optim.Optimizer
StateDict = Dict[str, Any]
Device = torch.device

def compute_level_masks(level_bits):
    code_len = sum(level_bits)        
    level_masks = []
    ones = 0
    for bits in level_bits:
        ones += bits
        zeros = code_len - ones
        mask = '0b' + ones * '1' + zeros * '0'
        mask = int(mask, 2) 
        level_masks.append(mask)
        # print(bin(mask))
    return level_masks
            

def compute_parent(bin_code: int, level: int, level_masks: List[int]) -> int:
    # print('level:', level)
    if level >= 2:
        parent = bin_code & level_masks[level-2]
    elif level == 1:
        parent = 0
    elif level == 0:
        parent = 0
    else:
        parent = -1
        print("Unknown parent")
    return parent


def compute_simmat(bin_codes: ArrayLike, levels: ArrayLike, level_masks: List[int], as_numpy: bool = True) -> Union[NDArray[np.int64], Tensor]:
    bin_codes = torch.LongTensor(bin_codes)
    levels = torch.LongTensor(levels)
    masks = torch.LongTensor(level_masks).reshape(1,1,-1)
    
    with torch.no_grad():
        lc_ancestors = bin_codes.bitwise_xor(bin_codes.view(-1,1)).bitwise_not()
        #print('ancestors.shape:', lc_ancestors.shape)        
        #print('ancestors:', lc_ancestors)

        lc_ancestors = lc_ancestors.unsqueeze(dim=-1)
        
        # levels for lc_ancestors
        simmat = (lc_ancestors.bitwise_and(masks) == masks).sum(-1) 
        #print('simmat.view:', lc_ancestors.bitwise_and(masks)[0,1,:])
        #print(simmat)        
        
        diag = torch.eye(len(levels), dtype=int) * levels.view(-1,1) - torch.eye(len(levels), dtype=int) * levels.max()
        #print('diag:\n', diag)                    
        simmat += diag
        
    if as_numpy:
        return simmat.numpy()

    return simmat


def build_dicts(simmat: NDArray[np.int64]) -> List[Dict[int, ArrayLike]]:
    dicts: List[Dict[int, ArrayLike]] = []
    for i in range(len(simmat)):
        labels: ArrayLike = np.arange(len(simmat))
        row = simmat[i, :]
        level_dict: Dict[int, ArrayLike] = {}
        for level in np.unique(row):
            level_dict[level] = labels[row == level]
        dicts.append(level_dict)
    return dicts
 

def bin_code2array(bin_code: int, length: int) -> ArrayLike:
    str_mask = bin(bin_code)[2:]
    # doplnenie nulami na zadanu dlzku
    str_mask = str_mask.rjust(length, '0')
    # premenna na pole 0 a 1
    bit_list = [ord(char) - ord('0') for char in str_mask]
    return np.array(bit_list)
    # arr = np.char.array(str_mask, unicode=False).view('int8')
    # arr = arr - ord('0')
    # return arr


def save_checkpoint(epoch: int, step: int, model: Network, optimizer: Optimizer, scheduler: Scheduler, path: str, verbose: bool = True):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_sched': None if scheduler is None else scheduler.state_dict()
        }

    torch.save(checkpoint, path)
    if verbose:
        print('save checkpoint:', epoch, step, flush=True)


def load_checkpoint(path: str, map_location: Device) -> Tuple[int, int, StateDict, StateDict, StateDict]:
    checkpoint = torch.load(path, map_location=map_location)
    epoch = checkpoint['epoch']
    step = checkpoint.get('step', 0)
    model_state = checkpoint['model']
    optimizer_state = checkpoint['optimizer']
    lr_sched_state = checkpoint['lr_sched']

    return epoch, step, model_state, optimizer_state, lr_sched_state


def load_model_from_checkpoint(path: str, map_location: Device) -> StateDict:
    checkpoint = torch.load(path, map_location=map_location)
    model_state = checkpoint['model']
    return model_state


