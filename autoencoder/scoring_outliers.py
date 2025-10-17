import time
import sys
import os

import logging
from typing import List, Any, Union
import numpy as np
import pandas as pd
from zipfile import ZipFile

from priprava_dat.utils import compute_odb

import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .autoencoder import DenoisingResnetAutoencoder
from .dataset import add_pe
from .hierarchy_loss import TreeAggregation, get_level_edges

from typing import Tuple, List, Dict
Tensor = torch.Tensor

# pomocna funkcia pre citanie z adresara alebo ZIP suboru
def open_file(model_dir, name: str):
    if isinstance(model_dir, ZipFile):
        file = model_dir.open(name)
    elif isinstance(model_dir, str):
        file = open(os.path.join(model_dir, name), 'rb')
    else:
        file = None
    return file

# trieda pre ciselniky
class CodeTables(object):
    def __init__(self, model_dir, path: str):
        # mapovacie subory
        map_prod_file = os.path.join(path,'mapovanie_produkty.csv')
        map_diag_file = os.path.join(path,'mapovanie_diagnozy.csv')
        map_odb_file = os.path.join(path,'mapovanie_odbornosti.csv')

        # citanie mapovania
        self.df_map_prod = pd.read_csv(open_file(model_dir, map_prod_file), usecols=['orig_kod_hier','novy_kod_hier'], dtype={'orig_kod_hier':np.uint64, 'novy_kod_hier':np.uint64})
        self.df_map_diag = pd.read_csv(open_file(model_dir, map_diag_file), usecols=['orig_kod_hier','novy_kod_hier'], dtype={'orig_kod_hier':np.uint32, 'novy_kod_hier':np.uint32})
        self.df_map_odb  = pd.read_csv(open_file(model_dir, map_odb_file), usecols=['orig_kod_hier','novy_kod_hier'], dtype={'orig_kod_hier':np.uint16, 'novy_kod_hier':np.uint16})

        self.df_map_prod.rename(columns={'orig_kod_hier' : 'id_prod', 'novy_kod_hier': 'id_prod_n'}, inplace=True)
        self.df_map_diag.rename(columns={'orig_kod_hier' : 'id_diag', 'novy_kod_hier': 'id_diag_n'}, inplace=True)
        self.df_map_odb.rename(columns={'orig_kod_hier' : 'id_odb_new', 'novy_kod_hier': 'id_odb_n'}, inplace=True)

        # ciselniky
        cs_prod_file = os.path.join(path,'le_produkty.csv')
        cs_diag_file = os.path.join(path,'le_diagnozy.csv')
        cs_odb_file = os.path.join(path,'le_odbornosti.csv')

        # nacitame le pre ciselniky
        # produkty
        self.df_cs_prod = pd.read_csv(open_file(model_dir, cs_prod_file), usecols=['kod_hier', 'label', 'uroven'],
                                dtype={'kod_hier': np.int64, 'label': np.int32, 'uroven': np.uint8})
        # diagnozy
        self.df_cs_diag = pd.read_csv(open_file(model_dir, cs_diag_file), usecols=['kod_hier', 'label', 'uroven'],
                                dtype={'kod_hier': np.int32, 'label': np.int16, 'uroven': np.uint8})
        # odbornosti
        self.df_cs_odb = pd.read_csv(open_file(model_dir, cs_odb_file), usecols=['kod_hier', 'label', 'uroven'],
                                dtype={'kod_hier': np.int16, 'label': np.uint8, 'uroven': np.uint8})

        self.df_cs_prod.sort_values(by='label', inplace=True)
        self.df_cs_diag.sort_values(by='label', inplace=True)
        self.df_cs_odb.sort_values(by='label', inplace=True)

        # doplnime Path Encoding
        cfg = Config()
        add_pe(self.df_cs_prod, level_bits=cfg.level_bits['id_prod_l'])
        add_pe(self.df_cs_diag, level_bits=cfg.level_bits['id_diag_l'])
        add_pe(self.df_cs_odb, level_bits=cfg.level_bits['id_odb_l'])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tree_agg_dg = TreeAggregation(get_level_edges(self.df_cs_diag, level_bits=cfg.level_bits['id_diag_l'], device=device)).to(device=device)
        self.tree_agg_prod = TreeAggregation(get_level_edges(self.df_cs_prod, level_bits=cfg.level_bits['id_prod_l'], device=device)).to(device=device)
        self.tree_agg_odb = TreeAggregation(get_level_edges(self.df_cs_odb, level_bits=cfg.level_bits['id_odb_l'], device=device)).to(device=device)

        # dodatocne premenovanie stlpcov
        self.df_cs_prod.rename(columns={'kod_hier': 'id_prod_b', 'label': 'id_prod_l', 'pe': 'pe_prod'}, inplace=True)
        self.df_cs_diag.rename(columns={'kod_hier': 'id_diag_b', 'label': 'id_diag_l', 'pe': 'pe_diag'}, inplace=True)
        self.df_cs_odb.rename(columns={'kod_hier': 'id_odb_b', 'label': 'id_odb_l', 'pe': 'pe_odb'}, inplace=True)

    # vrati pocty kategorii
    def get_n_cats(self) -> Dict[str, int]:
        n_cats: Dict[str, int] = {}
        n_cats['id_diag_l'] = self.df_cs_diag.id_diag_l.max() + 1
        n_cats['id_prod_l'] = self.df_cs_prod.id_prod_l.max() + 1
        n_cats['id_odb_l'] = self.df_cs_odb.id_odb_l.max() + 1
        n_cats['vek'] = 96 + 1 + 1
        n_cats['pohlavie'] = 3
        return n_cats

    # aplikovanie na vstupny subor
    def preprocess_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        # docasne vypnutie warningov
        pd.options.mode.chained_assignment = None

        # posun veku o 1
        input_df.loc[:, 'vek'] += 1

        # ciselne kodovanie pohlavia
        input_df.loc[:, 'pohlavie'] = input_df.pohlavie.fillna('U')
        input_df.loc[:, 'pohlavie'] = input_df.pohlavie.map({'M':1, 'Z':2, 'U':0})
        input_df.loc[:, 'pohlavie'] = input_df.pohlavie.astype(np.uint8)

        # derivovany stlpec id_odb_new
        new_col = input_df.apply(compute_odb, axis=1)
        input_df['id_odb_new'] = new_col
        input_df.loc[:, 'id_odb_new'] = input_df.id_odb_new.astype(np.uint16)

        # potrebne stlpce
        input_df = input_df.loc[:,['id_vzs','vek','pohlavie','id_prod','id_diag','id_odb_new']]

        # aplikovanie premapovania
        input_df = input_df.merge(self.df_map_prod, on='id_prod', how='left')
        idx = input_df.query("id_prod_n == id_prod_n", engine='python').index
        input_df.loc[idx, 'id_prod'] = input_df.loc[idx, 'id_prod_n']
        input_df.drop(columns=["id_prod_n"], inplace=True)

        input_df = input_df.merge(self.df_map_diag, on='id_diag', how='left')
        idx = input_df.query("id_diag_n == id_diag_n", engine='python').index
        input_df.loc[idx, 'id_diag'] = input_df.loc[idx, 'id_diag_n']
        input_df.drop(columns=["id_diag_n"], inplace=True)

        input_df = input_df.merge(self.df_map_odb, on='id_odb_new', how='left')
        idx = input_df.query("id_odb_n == id_odb_n", engine='python').index
        input_df.loc[idx, 'id_odb_new'] = input_df.loc[idx, 'id_odb_n']
        input_df.drop(columns=["id_odb_n"], inplace=True)
        # print('Mapping applied')

        # aplikovanie label encodingu
        input_df.rename(columns={'id_odb_new':'id_odb_b','id_diag':'id_diag_b','id_prod':'id_prod_b'}, inplace=True)
        input_df = input_df.merge(self.df_cs_prod, on='id_prod_b', how='inner')
        input_df = input_df.merge(self.df_cs_diag, on='id_diag_b', how='inner')
        input_df = input_df.merge(self.df_cs_odb, on='id_odb_b', how='inner')
        # print('Code tables applied')
        return input_df


# specialny dataset pre scoring
class ScoringDataset(Dataset[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                              Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, input_df: pd.DataFrame):

        #input_df = code_tables.preprocess_data(input_df)

        # premenne
        self.diag_b = np.array(input_df.id_diag_b.values, dtype=np.uint32)
        self.diag_pe = input_df.pe_diag.values
        self.prod_b = np.array(input_df.id_prod_b.values, dtype=np.uint64)
        self.prod_pe = input_df.pe_prod.values
        self.odb_b = np.array(input_df.id_odb_b.values, dtype=np.uint16)
        self.odb_pe = input_df.pe_odb.values
        self.gender = np.array(input_df.pohlavie.values, dtype=np.uint8)
        self.age = np.array(input_df.vek.values, dtype=np.uint8)

        # LE vstupy
        self.diag_le = np.array(input_df.id_diag_l.values, dtype=np.uint16)
        self.prod_le = np.array(input_df.id_prod_l.values, dtype=np.uint16)
        self.odb_le = np.array(input_df.id_odb_l.values, dtype=np.uint16)

    def __len__(self):
        return len(self.diag_b)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                           Tensor, Tensor, Tensor, Tensor]:

        diag_b = self.diag_b[i]
        diag_pe = self.diag_pe[i]
        prod_b = self.prod_b[i]
        prod_pe = self.prod_pe[i]
        odb_b = self.odb_b[i]
        odb_pe = self.odb_pe[i]
        gender = self.gender[i]
        age = self.age[i]

        diag_le = self.diag_le[i]
        prod_le = self.prod_le[i]
        odb_le = self.odb_le[i]

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

# trieda pre vypocet liftu
class Scoring(object):

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models = []
        self.freq_df = None
        self.code_tables = None

    def load_model(self, model_path: str):
        self.logger.info('Load model: ' + model_path)

        if os.path.isdir(model_path):
            model_dir = model_path
        else:
            model_dir = ZipFile(model_path, 'r')

        # nacitanie frekvencneho suboru
        self.freq_df = pd.read_csv(open_file(model_dir, 'freqs.csv'))
        #print(len(freq_df))
        # celkovy pocet riadkov
        total_rows = self.freq_df.query("premenna == 'pohlavie'").pocet.sum()
        # previest na pravdepodobnosti
        self.freq_df['prob'] = self.freq_df.pocet / total_rows
        self.logger.info('Frequency file read: ' + str(len(self.freq_df)) + " rows")

        # load code tables
        self.code_tables = CodeTables(model_dir, 'ciselniky')
        self.logger.info('Code tables loaded')

        # load the model
        cfg = Config()
        n_cats = self.code_tables.get_n_cats()
        self.logger.info('N cats: ' + str(n_cats))

        # to device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # ci ide o zip file
        if isinstance(model_dir, ZipFile):
            files = model_dir.namelist()
        else:
            files = os.listdir(model_dir)

        # teraz pouzivame h5 subory
        h5_files = [f for f in files if f.endswith('.h5')]
        assert len(h5_files) > 0, "Nenasiel sa ziadny h5 subor s vahami modelu"

        for h5_file in h5_files:
            self.logger.info('h5 file: ' + h5_file)
            autoencoder = DenoisingResnetAutoencoder(n_cats=n_cats, emb_size=cfg.emb_size, level_bits=cfg.level_bits,
                                            row_emb_size=cfg.row_emb_size, bias=cfg.bias, hsize=cfg.hsize,
                                            blocks=cfg.n_res_blocks, initialization=cfg.initialization,
                                            norm=cfg.norm, norm_emb=cfg.norm_emb, dropout=cfg.dropout, activation=cfg.activation,
                                            stats_dict={}, mask_pct=cfg.mask_pct, noise_pct=cfg.noise_pct)

            autoencoder = autoencoder.to(device=device)

            # load from .h5
            h5_file = open_file(model_dir, h5_file)
            model_state = torch.load(h5_file, map_location=device)
            autoencoder.load_state_dict(model_state)

            # to eval mode
            autoencoder.eval()
            self.models.append(autoencoder)

        self.logger.info('Init completed')

    def process_file(self, inp_file: str, out_file: str, log_file: str) -> None:
        self.logger.debug('Processing file: ' + inp_file)

        with torch.no_grad():
            cfg = Config()

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            df = pd.read_csv(inp_file,
                            usecols=['id_vzs', 'vek', 'pohlavie', 'id_diag', 'id_prod', 'id_odb', 'id_odb_odos', 'id_odb_odporuc'],
                            dtype={'id_vzs': np.int32,
                                    'vek': np.int16,
                                    'id_odb': np.uint16,
                                    'id_odb_odos': np.uint16,
                                    'id_odb_odporuc': np.uint16,
                                    'id_prod':np.uint64,
                                    'id_diag':np.uint32})

            # najskor ocistime davku od riadkov s produktmi, na ktorych model nebol trenovany (iter1.1)
            df['l1_prod'] = df.id_prod.apply(lambda x: x >> 39)
            df['l2_prod'] = df.id_prod.apply(lambda x: x >> 31 & 0b11111111)
            #df['l3_prod'] = df.id_prod.apply(lambda x: x >> 23 & 0b11111111)
            q =  'l1_prod in [0b1100, 0b0111, 0b1010, 0b1011] and ' # Z,O,S,V
            q += '(l1_prod != 0b1011 or l2_prod in [0b00000101, 0b00000110]) ' # VS,VV
            df_filtered = df.query(q)
            df_filtered = self.code_tables.preprocess_data(df_filtered)
            self.logger.debug('len(df_filtered): ' + str(len(df_filtered)))
            not_q =  'l1_prod not in [0b1100, 0b0111, 0b1010, 0b1011] or ' # Z,O,S,V
            not_q += '(l1_prod == 0b1011 and l2_prod not in [0b00000101, 0b00000110]) ' # VS,VV
            df_rem = df.query(not_q)
            df_rem = self.code_tables.preprocess_data(df_rem)
            self.logger.debug('len(df_rem): ' + str(len(df_rem)))

            # vytvorime dataset nad tymto suborom
            ds = ScoringDataset(df_filtered)

            # data loader
            batch_size = cfg.batch_size
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

            age_list, gender_list, diag_list, prod_list, odb_list = [], [], [], [], []
            age_prob_list, gender_prob_list, diag_prob_list, prod_prob_list, odb_prob_list = [], [], [], [], []

            for diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, diag_le, prod_le, odb_le, age, gender in dl:
                #print('input:', diag_b)

                diag_b = diag_b.to(device=device)
                prod_b = prod_b.to(device=device)
                odb_b = odb_b.to(device=device)
                diag_pe = diag_pe.to(device=device)
                prod_pe = prod_pe.to(device=device)
                odb_pe = odb_pe.to(device=device)
                diag_le = diag_le.to(device=device)
                prod_le = prod_le.to(device=device)
                odb_le = odb_le.to(device=device)
                age = age.to(device=device)
                gender = gender.to(device=device)

                age_list += age.tolist()
                gender_list += gender.tolist()
                diag_list += diag_b.tolist()
                prod_list += prod_b.tolist()
                odb_list += odb_b.tolist()

                # predikcia - moze byt ensemble
                age_probs = []
                gender_probs = []
                diag_probs = []
                prod_probs = []
                odb_probs = []

                for model in self.models:
                    diag_r, _, _, _, _ = model(diag_b*0, prod_b, odb_b, diag_pe*0, prod_pe, odb_pe, age, gender)
                    _, prod_r, _, _, _ = model(diag_b, prod_b*0, odb_b, diag_pe, prod_pe*0, odb_pe, age, gender)
                    _, _, odb_r, _, _ = model(diag_b, prod_b, odb_b*0, diag_pe, prod_pe, odb_pe*0, age, gender)
                    _, _, _, age_r, _ = model(diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, age*0, gender)
                    _, _, _, _, gender_r = model(diag_b, prod_b, odb_b, diag_pe, prod_pe, odb_pe, age, gender*0)

                    age_prob = age_r.gather(1, age.view(-1,1)).squeeze()
                    age_probs.append(age_prob)

                    gender_prob = torch.softmax(gender_r, dim=1)
                    gender_prob = gender_prob.gather(1, gender.view(-1,1)).squeeze()
                    gender_probs.append(gender_prob)

                    diag_r_agg = self.code_tables.tree_agg_dg(diag_r)
                    diag_prob = diag_r_agg.gather(1, diag_le.view(-1,1)).squeeze()
                    diag_probs.append(diag_prob)

                    prod_r_agg = self.code_tables.tree_agg_prod(prod_r)
                    prod_prob = prod_r_agg.gather(1, prod_le.view(-1,1)).squeeze()
                    prod_probs.append(prod_prob)

                    odb_r_agg = self.code_tables.tree_agg_odb(odb_r)
                    odb_prob = odb_r_agg.gather(1, odb_le.view(-1,1)).squeeze()
                    odb_probs.append(odb_prob)

                if len(self.models) == 1:
                    age_prob = age_probs[0]
                    gender_prob = gender_probs[0]
                    diag_prob = diag_probs[0]
                    prod_prob = prod_probs[0]
                    odb_prob = odb_probs[0]
                else:
                    age_prob = sum(age_probs) / len(age_probs)
                    gender_prob = sum(gender_probs) / len(gender_probs)
                    diag_prob = sum(diag_probs) / len(diag_probs)
                    prod_prob = sum(prod_probs) / len(prod_probs)
                    odb_prob = sum(odb_probs) / len(odb_probs)

                age_prob_list += age_prob.tolist() if age_prob.numel() > 1 else [age_prob.item()]
                gender_prob_list += gender_prob.tolist() if gender_prob.numel() > 1 else [gender_prob.item()]
                diag_prob_list += diag_prob.tolist() if diag_prob.numel() > 1 else [diag_prob.item()]
                prod_prob_list += prod_prob.tolist() if prod_prob.numel() > 1 else [prod_prob.item()]
                odb_prob_list += odb_prob.tolist() if odb_prob.numel() > 1 else [odb_prob.item()]

            # print(df.dtypes)
            result = pd.DataFrame()
            result['id_vzs'] = df_filtered.id_vzs
            result['vek'] = age_list
            result['pohlavie'] = gender_list
            result['id_diag_b'] = diag_list
            result['id_prod_b'] = prod_list
            result['id_odb_b'] = odb_list
            result['vek_net_prob'] = age_prob_list
            result['pohlavie_net_prob'] = gender_prob_list
            result['id_diag_b_net_prob'] = diag_prob_list
            result['id_prod_b_net_prob'] = prod_prob_list
            result['id_odb_b_net_prob'] = odb_prob_list
            #print(result)

            for var, freq_df_part in self.freq_df.groupby('premenna'):
                #print('var', var)
                freq_df_part = freq_df_part.loc[:, ['hodnota', 'prob']]
                freq_df_part.rename(columns={'hodnota': var, 'prob': var + '_prob'}, inplace=True)
                result = result.merge(freq_df_part, on=var, how='left')
                result[var + '_lift'] = result[var + '_net_prob'] / result[var + '_prob']

            #print('df_rem:', df_rem)

            df_rem['vek_lift'] = np.nan
            df_rem['pohlavie_lift'] = np.nan
            df_rem['id_diag_b_lift'] = np.nan
            df_rem['id_prod_b_lift'] = np.nan
            df_rem['id_odb_b_lift'] = np.nan
            result = pd.concat([result, df_rem])

            result = result.loc[:,['id_vzs','vek','pohlavie','id_diag_b','id_prod_b','id_odb_b',
            'vek_lift','pohlavie_lift','id_diag_b_lift', 'id_prod_b_lift', 'id_odb_b_lift']]
            result.to_csv(out_file, index=False)

if __name__ == '__main__':

    # construct scoring object
    logger = logging.getLogger(name=__name__)
    scoring = Scoring(logger)

    # loading model
    base_path = '/home/azureuser/localfiles/v1'
    model_path = base_path + '/models/autoencoder/model48_52'
    scoring.load_model(model_path)
    logger.info('Model loaded')

    # nacitanie vsetkych suborov z adresara a spustenie run()
    batch_path = '/home/azureuser/localfiles/v1/batch'
    batch_folder = batch_path + '/fa_9641279'
    files = os.listdir(batch_folder)

    # process files
    output_path = '/home/azureuser/localfiles/v1/output'
    output_folder = output_path + '/fa_9641279'
    os.makedirs(output_folder, exist_ok=True)
    for f in files:
        idx = f.rfind('.')
        if idx >= 0:
            out_f = f[0:idx] + '_out' + f[idx:]
            log_f = f[0:idx] + '.log'
        else:
            out_f = f + '.out'
            log_f = f + '.log'

        out_file = os.path.join(output_folder, out_f)
        log_file = os.path.join(output_folder, log_f)
        inp_file = os.path.join(batch_folder, f)

        # spracovanie suboru
        try:
            scoring.process_file(f, inp_file, out_file, log_file)
            logger.info('File: ' + f + ' processed.')
        except Exception as e:
            with open(log_file, "a") as log:
                print(type(e), file=log)
                print(e, file=log)
