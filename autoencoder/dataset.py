# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:23:12 2024

@author: bugatap
"""

import time

import pandas as pd
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .utils import compute_parent, compute_pe, compute_level_masks

from typing import Tuple, List
Tensor = torch.Tensor


def add_pe(df_cs: pd.DataFrame, level_bits: list[int]):
    df_cs.sort_values(by='kod_hier', inplace=True)

    levels = np.array(df_cs.uroven.values, dtype=np.int8)
    bin_codes = np.array(df_cs.kod_hier.values, dtype=np.int64)
    level_masks = compute_level_masks(level_bits)

    pe: List[List[int]] = []
    for i in range(len(bin_codes)):
        item_pe = compute_pe(bin_codes[i], levels[i], level_masks, bin_codes)
        pe.append(item_pe)

    df_cs['pe'] = pe


def read_code_tables(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ciselniky
    cs_prod_file = 'le_produkty.csv'
    cs_diag_file = 'le_diagnozy.csv'
    cs_odb_file = 'le_odbornosti.csv'

    # nacitame le pre ciselniky
    # produkty
    df_cs_prod = pd.read_csv(path + '/ciselniky/' + cs_prod_file, usecols=['kod_hier', 'label', 'uroven'],
                            dtype={'kod_hier': np.int64, 'label': np.int32, 'uroven': np.uint8})
    # diagnozy
    df_cs_diag = pd.read_csv(path + '/ciselniky/' + cs_diag_file, usecols=['kod_hier', 'label', 'uroven'],
                            dtype={'kod_hier': np.int32, 'label': np.int16, 'uroven': np.uint8})
    # odbornosti
    df_cs_odb = pd.read_csv(path + '/ciselniky/' + cs_odb_file, usecols=['kod_hier', 'label', 'uroven'],
                            dtype={'kod_hier': np.int16, 'label': np.uint8, 'uroven': np.uint8})

    df_cs_prod.sort_values(by='label', inplace=True)
    df_cs_diag.sort_values(by='label', inplace=True)
    df_cs_odb.sort_values(by='label', inplace=True)

    return df_cs_prod, df_cs_diag, df_cs_odb


def add_pe_from_code_tables(input_df: pd.DataFrame,
                            df_cs_prod: pd.DataFrame, df_cs_diag: pd.DataFrame, df_cs_odb: pd.DataFrame,
                            path: str, cfg: Config) -> pd.DataFrame:

    df_cs_prod_c = df_cs_prod.copy()
    df_cs_diag_c = df_cs_diag.copy()
    df_cs_odb_c = df_cs_odb.copy()

    # doplnime Path Encoding
    add_pe(df_cs_prod_c, level_bits=cfg.level_bits['id_prod_l'])
    add_pe(df_cs_diag_c, level_bits=cfg.level_bits['id_diag_l'])
    add_pe(df_cs_odb_c, level_bits=cfg.level_bits['id_odb_l'])

    # dodatocne premenovanie stlpcov
    df_cs_prod_c.rename(columns={'kod_hier': 'id_prod_b', 'label': 'id_prod_l', 'pe': 'pe_prod'}, inplace=True)
    df_cs_diag_c.rename(columns={'kod_hier': 'id_diag_b', 'label': 'id_diag_l', 'pe': 'pe_diag'}, inplace=True)
    df_cs_odb_c.rename(columns={'kod_hier': 'id_odb_b', 'label': 'id_odb_l', 'pe': 'pe_odb'}, inplace=True)

    # pripojime ciselniky
    input_df = input_df.merge(df_cs_prod_c, on='id_prod_l', how='inner')
    input_df = input_df.merge(df_cs_diag_c, on='id_diag_l', how='inner')
    input_df = input_df.merge(df_cs_odb_c, on='id_odb_l', how='inner')
    return input_df


class AEDataset(Dataset[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                              Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, input_df: pd.DataFrame):

        self.diag_b = np.array(input_df.id_diag_b.values, dtype=np.uint32)
        self.diag_pe = input_df.pe_diag.values
        self.prod_b = np.array(input_df.id_prod_b.values, dtype=np.uint64)
        self.prod_pe = input_df.pe_prod.values
        self.odb_b = np.array(input_df.id_odb_b.values, dtype=np.uint16)
        self.odb_pe = input_df.pe_odb.values
        self.gender = np.array(input_df.pohlavie.values, dtype=np.uint8)
        self.age = np.array(input_df.vek.values, dtype=np.uint8)

        # LE vstupy - len kvoli loss function
        self.diag_le = np.array(input_df.id_diag_l.values, dtype=np.uint16)
        self.prod_le = np.array(input_df.id_prod_l.values, dtype=np.uint16)
        self.odb_le = np.array(input_df.id_odb_l.values, dtype=np.uint16)

        # pocty
        self.cnts = np.array(input_df.pocet.values, dtype=np.int64)

        # vytvorime velke pole smernikov do dat (indexov)
        self.idxs = np.repeat(np.arange(len(input_df)), self.cnts)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                           Tensor, Tensor, Tensor, Tensor]:
        j = self.idxs[i]

        diag_b = self.diag_b[j]
        diag_pe = self.diag_pe[j]
        prod_b = self.prod_b[j]
        prod_pe = self.prod_pe[j]
        odb_b = self.odb_b[j]
        odb_pe = self.odb_pe[j]
        gender = self.gender[j]
        age = self.age[j]

        diag_le = self.diag_le[j]
        prod_le = self.prod_le[j]
        odb_le = self.odb_le[j]

        diag_b = torch.as_tensor(diag_b, dtype=torch.long)
        diag_pe = torch.as_tensor(diag_pe, dtype=torch.long)
        prod_b = torch.as_tensor(prod_b, dtype=torch.long)
        prod_pe = torch.as_tensor(prod_pe, dtype=torch.long)
        odb_b = torch.as_tensor(odb_b, dtype=torch.long)
        odb_pe = torch.as_tensor(odb_pe, dtype=torch.long)
        gender = torch.as_tensor(gender, dtype=torch.long)
        age = torch.as_tensor(age, dtype=torch.long)

        diag_le = torch.as_tensor(diag_le, dtype=torch.long)
        prod_le = torch.as_tensor(prod_le, dtype=torch.long)
        odb_le = torch.as_tensor(odb_le, dtype=torch.long)

        return diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, diag_le, prod_le, odb_le, age, gender


def test_loader(dl: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                     Tensor, Tensor, Tensor]],
                n_epochs: int = 2, verbose: bool = False):
    times = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        t1 = time.time()
        for diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, diag_le, prod_le, odb_le, age, gender in dl:
            if verbose:
                print('diag_b.shape:', diag_b.shape, 'diag_pe.shape:', diag_pe.shape, 'prod_pe.shape:', prod_pe.shape,
                      'odb_pe.shape:', odb_pe.shape)
        t2 = time.time()
        times[epoch] = t2 - t1
    print('Average epoch duration:', times.mean())


def read_data(data_path: str, cfg: Config, pe_flag: bool=True, test_set: bool=False) \
    -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # vstupne cesty a mena suborov
    path = data_path
    if test_set:
        input_file = 'vzs_train_t.parquet'
    else:
        input_file = 'vzs_train.parquet'

    # hranica pocetnosti
    th_count = cfg.max_row_count

    # nacitanie ciselnikov
    df_cs_prod, df_cs_diag, df_cs_odb = read_code_tables(data_path)

    # vstupny subor
    input_df = pd.read_parquet(path + '/train/' + input_file, engine='fastparquet',
                               columns=['id_prod_l', 'id_diag_l', 'id_odb_l', 'pohlavie', 'vek', 'pocet'])

    # add PE if needed
    if pe_flag:
        input_df = add_pe_from_code_tables(input_df, df_cs_prod, df_cs_diag, df_cs_odb,
                                           data_path, cfg)

    # orezanie pocetnosti
    if not test_set:
        input_df.loc[input_df.query("pocet > @th_count").index, 'pocet'] = th_count

    # osetrime zaporny vek (nezname odbobie narodenia)
    #input_df.loc[input_df.query("vek < 0").index, 'vek'] = 0
    input_df.vek += 1
    return input_df, df_cs_prod, df_cs_diag, df_cs_odb


if __name__ == '__main__':
    # cesta k suborom
    base_path = '/home/azureuser/localfiles'

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi_data", type=str, help="path to poi data", default=base_path + '/input/azure_c_poi.parquet')
    parser.add_argument("--vzs_data", type=str, help="path to vzs data", default=base_path + '/input/azure_vzs.parquet')
    parser.add_argument("--data_path", type=str, help="path to data folder", default=base_path + '/data')
    parser.add_argument("--models_path", type=str, help="path to models folder", default=base_path + '/models')
    args = parser.parse_args()

    data_path = args.data_path

    # config file
    cfg = Config()

    input_df = read_data(data_path, cfg)

    # pocetnosti
    cnts = np.array(input_df.pocet.values, dtype=np.int64)
    print('Total rows:', sum(cnts), flush=True)
    print('Unique rows:', len(cnts), flush=True)

    # construct dataset and loader
    ds = AEDataset(input_df)

    dl: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = \
        DataLoader(ds, batch_size=1024, num_workers=8, shuffle=True)
    test_loader(dl, n_epochs=2, verbose=True)
