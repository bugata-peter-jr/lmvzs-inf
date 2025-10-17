import sys
import os

import pandas as pd

import numpy as np
import torch

from typing import List

Tensor = torch.Tensor


class TreeAggregation(torch.nn.Module):
    def __init__(self, level_edges: List[Tensor]):
        super().__init__()
        self.level_edges = []
        for edges in level_edges:
            param = torch.nn.parameter.Parameter(data=edges, requires_grad=False)
            self.level_edges.append(param)

    def forward(self, x: Tensor) -> Tensor:
        # batch_size = x.size(0)

        for i in range(len(self.level_edges)-1, -1, -1):
            edges = self.level_edges[i]
            src = edges[0, :]
            dst = edges[1, :]

            values = x.index_select(dim=-1, index=src)
            x = x.index_add(dim=-1, index=dst, source=values)

            # zopakovanie indexov pre batch
            # src = src.repeat((batch_size,1))
            # dst = dst.repeat((batch_size,1))

            # values = x.gather(dim=-1, index=src)
            # x = x.scatter_add(dim=-1, index=dst, src=values)

        # orezanie nakolko kvoli zaokruhlovacim chybam moze vyjst hodnota viac ako 1.0
        # x = torch.clip(x, 0.0, 1.0)
        #assert x.min() >= 0.0, "Tree prob is " + str(x.min().item())
        #assert x.max() <= 1.0, "Tree prob is " + str(x.max().item())

        return x


class HierarchyLoss(torch.nn.Module):
    def __init__(self, levels: Tensor, base: float, coef: float, reduction: str='mean'):
        super().__init__()
        # self.levels = torch.nn.parameter.Parameter(data=levels, requires_grad=False)
        self.levels = levels
        self.n_levels = levels.max()
        self.base = base
        self.coef = coef
        self.ce_loss = torch.nn.NLLLoss(reduction=reduction)
        self.reduction = reduction
        self.eps = 1e-9

    def forward(self, probs: Tensor, target_pe: Tensor) -> Tensor:
        batch_size = probs.size(0)

        level_coef = 1.0
        if self.reduction == 'none':
            err = torch.zeros(batch_size, device=probs.device)
        else:
            err = torch.tensor(0.0, device=probs.device)

        probs = torch.clip(probs, self.eps, 1.0)
        log_probs = torch.log(probs)
        log_probs[:, 0] = 0.0
        #print('log_probs:', log_probs)

        for act_level in range(self.n_levels, 0, -1):
            #print('act_level:', act_level)

            level_target = target_pe[:, act_level]
            #print('level_target:', level_target)

            loss = self.ce_loss(log_probs, level_target)
            #print('loss:', loss)

            #print('coef:', coef)
            err += loss * level_coef
            level_coef *= self.base

        return err * self.coef


class Node:
    def __init__(self, code, parent, level, label):
        self.code = code        # binarny (hierarchicky) kod uzla
        self.parent = parent    # kod rodica alebo priamo smernik na rodicovsky uzol
        self.level = level      # uroven uzla v strome
        self.label = label      # label v LE
        self.children = []      # smerniky na uzly zodpovedajuce detom

    def add_child(self, child):
        self.children.append(child)

    def write_edges(self, edges):
        for c in self.children:
            edges.append( (self.level+1, c.label, self.label) )
            c.write_edges(edges)


def create_tree(df_cs: pd.DataFrame, level_bits) -> Node:

    # usporiadame podla urovne
    df_cs.sort_values(by=['uroven', 'kod_hier'], inplace=True)

    # vypocet bitovych masiek pre jednotlive urovne
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

    # vytvorenie stromu
    tree = {}
    for _, row in df_cs.iterrows():

        # vypocet kodu rodica
        code = row.kod_hier
        level = row.uroven
        label = row.label
        #print('level:', level)
        if level >= 2:
            parent = code & level_masks[level-2]
        elif level == 1:
            parent = 0
        else:
            parent = -1

        node = Node(code, parent, level, label)
        tree[code] = node

    root = tree[0]
    # nastavenie vztahu rodic - deti
    for _, node in tree.items():
        if node.parent < 0:
            node.parent = None
            continue
        #print('Node.code:', node.code, 'parent.code:', node.parent)
        parent = tree.get(node.parent)
        parent.add_child(node)
        node.parent = parent

    return root


def get_level_edges(df_cs: pd.DataFrame, level_bits: List[int], device) -> List[Tensor] :

    root = create_tree(df_cs, level_bits)

    edges = []
    root.write_edges(edges)
    #print(edges)

    n_levels = len(level_bits)

    level_src = [[] for _ in range(n_levels)]
    level_dst = [[] for _ in range(n_levels)]

    for edge in edges:
        level, src, dst = edge
        level_src[level-1].append(src)
        level_dst[level-1].append(dst)

    level_edges = []
    for i in range(n_levels):
        t = torch.LongTensor([level_src[i], level_dst[i]]).to(device=device)
        level_edges.append(t)

    return level_edges

class AgeHierarchicalLoss(torch.nn.Module):
    def __init__(self, age_thresholds:List[List[int]], base:float, coef:float, apply_age_range:bool=False, reduction: str='mean'):
        super().__init__()

        age_thresholds_a = np.array(age_thresholds)
        age_thresholds_a = age_thresholds_a[:,1:]
        age_thresholds_a = age_thresholds_a + 1    # lebo v config je uvedeny vek v rokoch a label je o 1 vacsi

        # vekova skupina
        age_group = np.zeros(shape=(98,98))
        # vekova nadkategoria
        age_supercat = np.zeros(shape=(98,98))
        # vekove pasmo
        age_range = np.zeros(shape=(98,98))

        for i in range(len(age_thresholds_a)):
            l2_from, l2_to, l1_from, l1_to, l0_from, l0_to = age_thresholds_a[i]
            age_group[i,l2_from:l2_to+1] = 1
            age_supercat[i,l1_from:l1_to+1] = 1
            age_range[i,l0_from:l0_to+1] = 1

        self.age_group = torch.nn.parameter.Parameter(torch.Tensor(age_group), requires_grad=False)
        self.age_supercat = torch.nn.parameter.Parameter(torch.Tensor(age_supercat), requires_grad=False)
        self.age_range = torch.nn.parameter.Parameter(torch.Tensor(age_range), requires_grad=False)
        self.base = base
        self.coef = coef
        self.apply_age_range = apply_age_range
        self.eps = 1e-9

        # loss for levels
        self.l3_loss = torch.nn.NLLLoss(reduction=reduction)
        self.l2_loss = torch.nn.BCELoss(reduction=reduction)
        self.l1_loss = torch.nn.BCELoss(reduction=reduction)
        if self.apply_age_range:
            self.l0_loss = torch.nn.BCELoss(reduction=reduction)

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:

        log_probs = torch.log(probs)
        l3_loss = self.l3_loss(log_probs, target)

        target_bin = torch.ones_like(target).float()
        probs_for_l2 = (probs * self.age_group[target]).sum(axis=-1)
        probs_for_l2 = torch.clip(probs_for_l2, self.eps, 1.0)
        l2_loss = self.l2_loss(probs_for_l2, target_bin)

        probs_for_l1 = (probs * self.age_supercat[target]).sum(axis=-1)
        probs_for_l1 = torch.clip(probs_for_l1, self.eps, 1.0)
        l1_loss = self.l1_loss(probs_for_l1, target_bin)

        if self.apply_age_range:
            probs_for_l0 = (probs * self.age_range[target]).sum(axis=-1)
            probs_for_l0 = torch.clip(probs_for_l0, self.eps, 1.0)
            l0_loss = self.l1_loss(probs_for_l0, target_bin)

        total_loss = l3_loss + self.base * l2_loss + self.base**2 * l1_loss

        if self.apply_age_range:
            total_loss = total_loss + self.base**2 * l0_loss

        return self.coef * total_loss

if __name__ == '__main__':

    '''
    # test vypoctu hierarchickej loss
    probs = torch.Tensor([[0, 0, 0, 0.5, 0.1, 0.2, 0.2],
                          [0, 0, 0.5, 0.5, 0, 0, 0]])
    print('probs.shape:', probs.shape)

    level_edges1 = torch.LongTensor([[1,2],[0,0]])
    level_edges2 = torch.LongTensor([[3,4],[1,1]])
    level_edges3 = torch.LongTensor([[5,6],[4,4]])

    level_edges = [level_edges1, level_edges2, level_edges3]

    # agregacia pravdepodobnosti v strome
    tree_agg = TreeAggregation((level_edges))
    probs = tree_agg(probs)
    print(probs)

    # vypocet hierarchickej loss
    levels = torch.LongTensor([0,1,1,2,2,3,3])
    base = 2.0
    loss = HierarchyLoss(levels, base, coef=1.0, reduction='none')

    target_pe = torch.LongTensor([[0,1,4,5], [0,2,0,0]])

    err = loss(probs, target_pe)
    print('err:', err)


    from autoencoder.embedding import PathEncoding
    from autoencoder.utils import add_pe

    # test vytvorenia level_edges pre strom

    # cesta k suborom
    base_path = '/home/azureuser/localfiles/v1'
    # base_path = '/projects/LMVZS'

    data_path = base_path + '/data'

    # load code tables
    cs_odb_file = 'le_odbornosti.csv'

    # produkty
    df_cs = pd.read_csv(data_path + '/ciselniky/' + cs_odb_file, usecols=['kod_hier', 'label', 'uroven'],
                        dtype={'kod_hier': np.int64, 'label': np.int32, 'uroven': np.uint8})

    level_bits=[6, 5, 4]
    n_levels = len(level_bits)

    add_pe(df_cs, level_bits)

    path_enc = PathEncoding(len(df_cs))

    for _, row in df_cs.head().iterrows():
        pe = torch.LongTensor(row.pe).unsqueeze(0)
        print(row.pe, path_enc(pe))


    level_edges = get_level_edges(df_cs, level_bits, device='cpu')
    print(level_edges)
    '''

    from config import Config
    cfg = Config()
    print(cfg)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    loss_age_coef = cfg.loss_coefs['vek']
    apply_age_range = True
    loss_fn_age = AgeHierarchicalLoss(age_thresholds=cfg.age_thresholds, base=cfg.hierarchy_loss_base,
                                      coef=loss_age_coef, apply_age_range=apply_age_range, reduction='none').to(device=device)

    preds = np.zeros(shape=(1, 98))
    preds[0, 1+1] = 0.1
    preds[0, 2+1] = 0.7
    preds[0, 3+1] = 0.1
    preds[0, 4+1] = 0.1

    output = torch.Tensor(preds, device=device)
    target = torch.Tensor([2+1], device=device).long()

    loss = loss_fn_age(output, target)
    print('loss:', loss)
