# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:46:54 2024

@author: bugatap
"""

import pandas as pd
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

# vypocet odb_new
def compute_odb(row: pd.Series) -> int:
    #odb_vyn = 0b100111  # L1 pre vynechavany podstrom
    #odb_posun_l1 = 9
    
    # odb_vyn_l = [0, 68, 69, 163, 1223, 641, 642, 643, 644, 645, 646]
    odb_vyn_l = [0, 100, 101, 195, 673, 674, 675, 676, 677, 678, 1286]    
    
    # 68 centrálne operačné sály                                                                           
    # 69 centrálna sterilizácia                                                                            
    # 163 preprava biologického materiálu
    # 1223 zubná technika
    # 641-646 LEKVYD 

    id_odb = row.id_odb
    if id_odb not in odb_vyn_l:
        return id_odb

    id_odb = row.id_odb_odos
    if id_odb not in odb_vyn_l:
        return id_odb

    id_odb = row.id_odb_odporuc
    if id_odb not in odb_vyn_l:
        return id_odb

    return 0


# vypocet veku z datumu narodenia
def compute_age_from_btime(birth_time_key: int, day_of_proc: int) -> int:
    y = birth_time_key // 100
    m = birth_time_key % 100
    d = 15
    birth_date = date(y, m, d)
    date_of_proc = date(2015, 1, 1) + relativedelta(days=day_of_proc)
    age = (date_of_proc - birth_date).days // 365
    # starsi ako 96 rokov budu tvorit samostatnu skupinu
    if age > 96:
        age = 96
    return age


# vypocet veku
def compute_age(row: pd.Series) -> int:
    if row.obdobie_narod is None or np.isnan(row.obdobie_narod):
        return -1

    birth_time_key = int(row.obdobie_narod)
    age = compute_age_from_btime(birth_time_key, int(row.dat_vykon))
    return age


# pomocna funkcia na urcenie poctu dni z timedelty
def get_days(arg) -> int:
    return pd.Timedelta(arg).days


# prevod na datum
def convert_to_date(row: pd.Series) -> int:
    dat_vykon = row.dat_vykon
    start_date = row.start_date
    dat_vykon = date(2015, 1, 1) + relativedelta(days=dat_vykon)
    return (dat_vykon - start_date).days