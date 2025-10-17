# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from pytorch_model_summary import summary  # type: ignore

from .config import Config
from .dataset import AEDataset, read_data
from .utils import Statistics, StatisticsExt, compute_stats
from .embedding import RowEmbedding
from .output import RowOutput

from typing import Tuple, Callable, Dict, List
Tensor = torch.Tensor


class Masking(torch.nn.Module):
    def __init__(self, mask_pct: float = 0.8, noise_pct: float = 0.1, stats_dict: Dict[str, Statistics] = {}):
        super().__init__()
        self.mask_pct = mask_pct
        self.noise_pct = noise_pct
        self.stats_dict = stats_dict

    def forward(self, x_dg_bin: Tensor, x_prod_bin: Tensor, x_odb_bin: Tensor,
                x_dg_pe: Tensor, x_prod_pe: Tensor, x_odb_pe: Tensor,
                x_age: Tensor, x_gender: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor,
                                                          Tensor, Tensor, Tensor, Tensor]:

        if not self.training:
            return x_dg_bin, x_prod_bin, x_odb_bin, x_dg_pe, x_prod_pe, x_odb_pe, x_age, x_gender

        bs = x_dg_bin.size(0)
        n_masked = int(self.mask_pct * bs)
        n_noised = int(self.noise_pct * bs)

        x_dg_bin_n, x_prod_bin_n, x_odb_bin_n = 1 * x_dg_bin, 1 * x_prod_bin, 1 * x_odb_bin
        x_dg_pe_n, x_prod_pe_n, x_odb_pe_n = 1 * x_dg_pe, 1 * x_prod_pe, 1 * x_odb_pe
        x_age_n, x_gender_n = 1 * x_age, 1 * x_gender    # kopia!!!

        # indexy premennych pre maskovanie
        j = torch.randint(0, 5, size=(n_masked,), device=x_age.device)
        # doplnime -1 na nemaskovane pozicie
        j = torch.concat([j, (-1) * torch.ones(bs - n_masked, dtype=torch.int, device=x_age.device)])

        # indexy premennych pre zasumovanie
        k = torch.randint(0, 5, size=(n_noised,), device=x_age.device)
        # doplnime -1 na nezasumovane pozicie
        k = torch.cat([(-1) * torch.ones(n_masked, dtype=torch.int, device=x_age.device), k, (-1) *
                       torch.ones(bs - n_masked - n_noised, dtype=torch.int, device=x_age.device)])

        # dg - nulovat bin aj PE sucasne
        idx_mask_dg = (j == 0)
        x_dg_bin_n[idx_mask_dg] = 0
        x_dg_pe_n[idx_mask_dg] = 0
        if self.noise_pct > 0:
            # dg - zasumovat bin aj PE sucasne
            idx_noise_dg = (k == 0)
            dg_stat = self.stats_dict['id_diag_l']
            assert isinstance(dg_stat, StatisticsExt)
            p_dg = dg_stat.probs
            # print(p_dg)
            n = torch.sum(idx_noise_dg).item()
            if n > 0:
                assert isinstance(n, int)
                mod_idx = torch.multinomial(p_dg, n, replacement=True).to(device=x_age.device)
                x_dg_bin_n[idx_noise_dg] = dg_stat.bin_codes[mod_idx]
                x_dg_pe_n[idx_noise_dg] = dg_stat.pe[mod_idx]

        # produkt - nulovat bin aj PE sucasne
        idx_mask_prod = (j == 1)
        x_prod_bin_n[idx_mask_prod] = 0
        x_prod_pe_n[idx_mask_prod] = 0
        if self.noise_pct > 0:
            # produkt - zasumovat bin aj PE sucasne
            idx_noise_prod = (k == 1)
            prod_stat = self.stats_dict['id_prod_l']
            assert isinstance(prod_stat, StatisticsExt)
            p_prod = prod_stat.probs
            # print(p_prod)
            n = torch.sum(idx_noise_prod).item()
            if n > 0:
                assert isinstance(n, int)
                mod_idx = torch.multinomial(p_prod, n, replacement=True).to(device=x_age.device)
                x_prod_bin_n[idx_noise_prod] = prod_stat.bin_codes[mod_idx]
                x_prod_pe_n[idx_noise_prod] = prod_stat.pe[mod_idx]

        # odbornost - nulovat bin aj PE sucasne
        idx_mask_odb = (j == 2)
        x_odb_bin_n[idx_mask_odb] = 0
        x_odb_pe_n[idx_mask_odb] = 0
        if self.noise_pct > 0:
            # odbornost - zasumovat bin aj PE sucasne
            idx_noise_odb = (k == 2)
            odb_stat = self.stats_dict['id_odb_l']
            assert isinstance(odb_stat, StatisticsExt)
            p_odb = odb_stat.probs
            # print(p_odb)
            n = torch.sum(idx_noise_odb).item()
            if n > 0:
                assert isinstance(n, int)
                mod_idx = torch.multinomial(p_odb, n, replacement=True).to(device=x_age.device)
                x_odb_bin_n[idx_noise_odb] = odb_stat.bin_codes[mod_idx]
                x_odb_pe_n[idx_noise_odb] = odb_stat.pe[mod_idx]

        # vek
        idx_mask_age = (j == 3)
        x_age_n[idx_mask_age] = 0
        if self.noise_pct > 0:
            idx_noise_age = (k == 3)
            age_stat = self.stats_dict['vek']
            labels_age = age_stat.labels
            p_age = age_stat.probs
            # print(p_age)
            n = torch.sum(idx_noise_age, dtype=torch.int32).item()
            if n > 0:
                assert isinstance(n, int)
                mod_vals_idx = torch.multinomial(p_age, n, replacement=True).to(device=x_age.device)
                x_age_n[idx_noise_age] = labels_age[mod_vals_idx]

        # pohlavie
        idx_mask_gender = (j == 4)
        x_gender_n[idx_mask_gender] = 0
        if self.noise_pct > 0:
            idx_noise_gender = (k == 4)
            gender_stat = self.stats_dict['pohlavie']
            labels_gender = gender_stat.labels
            p_gender = gender_stat.probs
            # print(p_gender)
            n = torch.sum(idx_noise_gender).item()
            if n > 0:
                assert isinstance(n, int)
                mod_vals_idx = torch.multinomial(p_gender, n, replacement=True).to(device=x_gender.device)
                x_gender_n[idx_noise_gender] = labels_gender[mod_vals_idx]

        return x_dg_bin_n, x_prod_bin_n, x_odb_bin_n, x_dg_pe_n, x_prod_pe_n, x_odb_pe_n, x_age_n, x_gender_n


# ResNet
class Block(torch.nn.Module):
    """The main building block of `ResNet`."""

    def __init__(self, dim_in: int, dim_expand: int, activation: Callable[[], torch.nn.Module] = torch.nn.ReLU, dropout: float = 0.0, norm: bool = False) -> None:
        super().__init__()
        self.norm = None
        if norm:
            self.norm = torch.nn.LayerNorm(dim_in)
        self.linear_first = torch.nn.Linear(dim_in, dim_expand, bias=True)
        self.activation = activation()
        self.linear_second = torch.nn.Linear(dim_expand, dim_in, bias=True)
        self.dropout = None
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x:Tensor) -> Tensor:
        x_input = x
        if self.norm is not None:
            x = self.norm(x)
        x = self.linear_first(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear_second(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x_input + x
        return x

class DenseResNet(torch.nn.Module):
    def __init__(self, dim_in: int, dim_expand: int, activation: Callable[[], torch.nn.Module] = torch.nn.ReLU,
                 dropout: float = 0.0, norm: bool = False, blocks: int = 3) -> None:
        super().__init__()

        self.blocks = torch.nn.Sequential(*[Block(dim_in, dim_expand, activation, dropout, norm) for _ in range(blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

# resnet autoencoder
class ResnetAutoencoder(torch.nn.Module):
    def __init__(self, n_cats: Dict[str, int], emb_size: Dict[str, int], level_bits: Dict[str, List[int]],
                 row_emb_size:int, bias: bool, hsize: int, activation: Callable[[], torch.nn.Module] = torch.nn.ReLU,
                 dropout: float = 0.0, norm: bool = False, norm_emb: bool = False, has_proj: bool = True,
                 blocks: int = 3, initialization: str = 'none'):

        super().__init__()

        self.emb = RowEmbedding(n_cats=n_cats, emb_size=emb_size, level_bits=level_bits,
                                row_emb_size=row_emb_size, initialization=initialization, bias=bias,
                                norm_emb = norm_emb)
        # print('row emb size:', row_emb_size)
        # print('hsize:', hsize)
        self.resnet = DenseResNet(dim_in=row_emb_size, dim_expand=hsize,
                                  dropout=dropout, norm=norm,
                                  activation=activation, blocks=blocks)

        self.ln_final = torch.nn.LayerNorm(row_emb_size)

        self.output = RowOutput(hsize=row_emb_size, n_cats=n_cats, emb_size=emb_size, activation=activation, has_proj=has_proj, hier_age=True)

    def forward(self, x_dg_bin: Tensor, x_prod_bin: Tensor, x_odb_bin: Tensor,
                x_dg_pe: Tensor, x_prod_pe: Tensor, x_odb_pe: Tensor,
                x_age: Tensor, x_gender: Tensor) -> Tensor:
        # print(type(x))
        emb = self.emb(x_dg_bin, x_prod_bin, x_odb_bin,
                       x_dg_pe, x_prod_pe, x_odb_pe,
                       x_age, x_gender)
        # print('emb:', emb.shape)
        res = self.resnet(emb)
        # print('res:', res.shape)
        res = self.ln_final(res)
        # print('res:', res.shape)
        out = self.output(res)
        return out


# denoising resnet autoencoder
class DenoisingResnetAutoencoder(ResnetAutoencoder):
    def __init__(self, n_cats: Dict[str, int], emb_size: Dict[str, int], level_bits: Dict[str, List[int]],
                 row_emb_size:int, bias: bool, hsize: int, activation: Callable[[], torch.nn.Module] = torch.nn.ReLU,
                 dropout: float = 0.0, norm: bool = False, norm_emb: bool = False, has_proj: bool = True,
                 blocks: int = 3, initialization: str = 'none',
                 stats_dict: Dict[str, Statistics] = {},
                 mask_pct: float = 0.8, noise_pct: float = 0.1):

        super().__init__(n_cats=n_cats, emb_size=emb_size, level_bits=level_bits,
                 row_emb_size=row_emb_size, bias=bias, hsize=hsize,
                 activation=activation, dropout=dropout, norm=norm, norm_emb=norm_emb, has_proj=has_proj,
                 blocks=blocks, initialization=initialization)

        self.masking = Masking(stats_dict=stats_dict, mask_pct=mask_pct, noise_pct=noise_pct)

    def forward(self, x_dg_bin: Tensor, x_prod_bin: Tensor, x_odb_bin: Tensor,
                x_dg_pe: Tensor, x_prod_pe: Tensor, x_odb_pe: Tensor,
                x_age: Tensor, x_gender: Tensor) -> Tensor:

        with torch.no_grad():
            x_dg_bin, x_prod_bin, x_odb_bin, x_dg_pe, x_prod_pe, x_odb_pe, x_age, x_gender = self.masking(
                x_dg_bin, x_prod_bin, x_odb_bin, x_dg_pe, x_prod_pe, x_odb_pe, x_age, x_gender)

        return super().forward(x_dg_bin, x_prod_bin, x_odb_bin,
                               x_dg_pe, x_prod_pe, x_odb_pe,
                               x_age, x_gender)

if __name__ == '__main__':
    # cesta k suborom
    base_path = '/home/azureuser/localfiles/v1'

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to data folder", default=base_path + '/data')
    parser.add_argument("--models_path", type=str, help="path to models folder", default=base_path + '/models')
    args = parser.parse_args()

    data_path = args.data_path

    # config file
    cfg = Config()

    # vstupny subor
    input_df, _, _, _ = read_data(data_path, cfg)

    # pocetnosti
    cnts = np.array(input_df.pocet.values, dtype=np.int64)
    print('Total rows:', sum(cnts), flush=True)
    print('Unique rows:', len(cnts), flush=True)

    # determine number of categories
    n_cats: Dict[str, int] = {cat_var: input_df[cat_var].max()+1 for cat_var in cfg.features}
    print('N categories:', n_cats)

    device = torch.device('cpu')
    if cfg.mask_pct > 0 or cfg.noise_pct > 0:
        stats_dict = compute_stats(input_df, cfg.features, device)
        # print('Stats computed.')
        autoencoder = DenoisingResnetAutoencoder(n_cats=n_cats, emb_size=cfg.emb_size, level_bits=cfg.level_bits,
                                            row_emb_size=cfg.row_emb_size, bias=cfg.bias, hsize=cfg.hsize,
                                            blocks=cfg.n_res_blocks, initialization=cfg.initialization,
                                            norm=cfg.norm, norm_emb=cfg.norm_emb, has_proj=cfg.output_proj,
                                            dropout=cfg.dropout, activation=cfg.activation,
                                            stats_dict=stats_dict, mask_pct=cfg.mask_pct, noise_pct=cfg.noise_pct)
    else:
        autoencoder = ResnetAutoencoder(n_cats=n_cats, emb_size=cfg.emb_size, level_bits=cfg.level_bits,
                        row_emb_size=cfg.row_emb_size, bias=cfg.bias, hsize=cfg.hsize,
                        norm=cfg.norm, norm_emb=cfg.norm_emb, has_proj=cfg.output_proj,
                        dropout=cfg.dropout, activation=cfg.activation,
                        blocks=cfg.n_res_blocks, initialization=cfg.initialization)

    # construct dataset
    ds = AEDataset(input_df.head())
    dl: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = \
        DataLoader(ds, batch_size=2, num_workers=0, shuffle=True)

    # summary
    for diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, diag_le, prod_le, odb_le, age, gender in dl:
        summary(autoencoder, diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, age, gender,
                show_hierarchical=True, print_summary=True, show_parent_layers=True, max_depth=None)  # type: ignore
        break
