# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:55:18 2023

@author: bugatap
"""

import torch
from inspect import getmembers

from typing import List, Any

def nested_list_to_str(nested_list: List[List[Any]]) -> str:
    s = ''
    for l in nested_list:
        s += str(l) + '\n'
    return s

class Config(object):

    def __init__(self):

        # max counts for rows
        self.max_row_count = 1000

        # features and encoding
        self.features = ['id_diag_l', 'id_prod_l', 'id_odb_l', 'vek', 'pohlavie']
        self.level_bits = {'id_prod_l': [4, 8, 8, 9, 5, 9], 'id_odb_l': [6, 5], 'id_diag_l': [5, 6, 6, 4, 4]}
        self.emb_size = {'id_prod_t': 32, 'id_prod_d': 32*2, 'id_odb_t': 16, 'id_odb_d': 16*2,
                         'id_diag_t': 32, 'id_diag_d': 32*2, 'vek_t': 16, 'vek_d':16, 'pohlavie': 8}
        # network config
        self.row_emb_size = 512
        self.hsize = self.row_emb_size * 4
        self.initialization = 'none'
        self.bias = True
        self.n_res_blocks = 12
        self.activation = torch.nn.GELU
        self.norm = True
        self.norm_emb = True
        self.output_proj = True
        self.dropout = 0.0

        # loss
        self.strict_hierarchy_loss = False
        self.hierarchy_loss_base = 1.0
        #self.loss_coefs = {'id_prod_l': 1.0/40, 'id_odb_l': 1.0, 'id_diag_l': 1.0/90, 'vek': 1.0/8}
        self.loss_coefs = {'id_prod_l': 1.0/2, 'id_odb_l': 1.0, 'id_diag_l': 1.0/4, 'vek': 1.0/3}
        self.apply_age_range = False
        self.strict_metric = False

        # loader config
        self.P = 7

        # whether to use masking autoencoder
        self.mask_pct = 0.80
        self.noise_pct = 0.10

        # path to embeddings
        self.emb_dir = '/embedding_hierarchie/model04'
        self.dg_file = 'hierarchy_d_emb.h5'
        self.prod_file = 'hierarchy_p_emb.h5'
        self.odb_file = 'hierarchy_o_emb.h5'
        self.age_file = 'hierarchy_v_emb.h5'

        # Train parameters
        self.n_epochs = 10                    # number of training epochs
        self.batch_size = 1024 * 4            # batch size
        self.optimizer_type = 'adamw'         # type of optimizer
        self.scheduler_type = 'one_cycle'     # type of learning rate scheduler
        self.max_lr = 0.0001                  # maximal learning rate
        self.anneal_strategy = 'linear'       # anneal strategy
        self.pct_start = 0.3                  # start pct
        self.div_final = 10000.0              # final div factor
        self.WD = 0.01                        # weight decay

        # verbose
        self.verbose = 2

        # batches to save checpoint
        self.save_steps = 18000

        self.display_steps = 10000
        # self.display_steps = 1

        # model
        self.model_name = 'model02'

        # age thresholds
        self.age_thresholds = [
                [-1,-1,-1,-1,-1,-1,-1],
                [0,0,0,0,2,0,1],
                [1,1,1,0,3,0,1],
                [2,2,2,0,4,2,5],
                [3,3,3,1,5,2,5],
                [4,4,4,2,7,2,5],
                [5,5,5,3,9,2,5],
                [6,6,7,4,11,6,9],
                [7,6,8,4,12,6,9],
                [8,7,9,5,13,6,9],
                [9,8,10,5,14,6,9],
                [10,9,11,5,15,10,17],
                [11,10,12,6,16,10,17],
                [12,11,13,7,17,10,17],
                [13,12,14,8,18,10,17],
                [14,13,15,9,20, 10,17],
                [15,14,16,10,22,10,17],
                [16,15,17,11,24,10,17],
                [17,16,18,12,26,10,17],
                [18,17,20,13,28,18,25],
                [19,18,21,14,29,18,25],
                [20,18,22,14,30,18,25],
                [21,19,23,15,31,18,25],
                [22,20,24,15,32,18,25],
                [23,21,25,16,33,18,25],
                [24,22,26,16,34,18,25],
                [25,23,27,17,35,18,25],
                [26,24,28,17,36,26,44],
                [27,25,29,18,37,26,44],
                [28,26,30,18,38,26,44],
                [29,27,31,19,39,26,44],
                [30,28,32,20,40,26,44],
                [31,29,33,21,41,26,44],
                [32,30,34,22,42,26,44],
                [33,31,35,23,43,26,44],
                [34,32,36,24,44,26,44],
                [35,33,37,25,45,26,44],
                [36,34,38,26,46,26,44],
                [37,35,39,27,47,26,44],
                [38,36,40,28,48,26,44],
                [39,37,41,29,49,26,44],
                [40,38,42,30,50,26,44],
                [41,39,43,31,51,26,44],
                [42,40,44,32,52,26,44],
                [43,41,45,33,53,26,44],
                [44,42,46,34,54,26,44],
                [45,43,47,35,55,45,64],
                [46,44,48,36,56,45,64],
                [47,45,49,37,57,45,64],
                [48,46,50,38,58,45,64],
                [49,47,51,39,59,45,64],
                [50,48,52,40,60,45,64],
                [51,49,53,41,61,45,64],
                [52,50,54,42,62,45,64],
                [53,51,55,43,63,45,64],
                [54,52,56,44,64,45,64],
                [55,53,57,45,65,45,64],
                [56,54,58,46,67,45,64],
                [57,55,59,47,69,45,64],
                [58,56,60,48,71,45,64],
                [59,57,61,49,73,45,64],
                [60,58,62,50,75,45,64],
                [61,59,63,51,77,45,64],
                [62,60,64,52,79,45,64],
                [63,61,65,53,81,45,64],
                [64,62,67,54,83,45,64],
                [65,63,69,55,85,65,79],
                [66,63,70,56,86,65,79],
                [67,64,71,56,87,65,79],
                [68,65,72,57,88,65,79],
                [69,65,73,57,89,65,79],
                [70,66,74,58,90,65,79],
                [71,67,75,58,91,65,79],
                [72,68,76,59,92,65,79],
                [73,69,77,59,93,65,79],
                [74,70,78,60,94,65,79],
                [75,71,79,60,95,65,79],
                [76,72,80,61,96,65,79],
                [77,73,81,61,96,65,79],
                [78,74,82,62,96,65,79],
                [79,75,83,62,96,65,79],
                [80,76,84,63,96,80,96],
                [81,77,85,63,96,80,96],
                [82,78,86,64,96,80,96],
                [83,79,87,64,96,80,96],
                [84,80,88,65,96,80,96],
                [85,81,89,65,96,80,96],
                [86,82,90,66,96,80,96],
                [87,83,91,67,96,80,96],
                [88,84,92,68,96,80,96],
                [89,85,93,69,96,80,96],
                [90,86,94,70,96,80,96],
                [91,87,95,71,96,80,96],
                [92,88,96,72,96,80,96],
                [93,89,96,73,96,80,96],
                [94,90,96,74,96,80,96],
                [95,91,96,75,96,80,96],
                [96,92,96,76,96,80,96]
        ]

    def __str__(self):
        s = ''
        for name, value in getmembers(self):
            if name.startswith('__'):
                continue
            if name == 'age_thresholds':
                s += name + ':'
                s += '\n'
                s += nested_list_to_str(value)
                s += '\n'
            else:
                s += name + ' : ' + str(value) + '\n'
        return s

if __name__ == '__main__':
    cfg = Config()
    print(cfg)
