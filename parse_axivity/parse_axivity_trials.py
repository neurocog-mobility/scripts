#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:23:25 2024

@author: abdulzaf
"""
import sys
import pandas as pd
import numpy as np
from openmovement.load import CwaData
from pathlib import Path

def read_axivity(DIR_DATA, file_cwa):
    # read axivity cwa file
    with CwaData(f'{DIR_DATA}/{file_cwa}', include_gyro=True, include_temperature=True, include_accel=True, include_light=True) as cwa_data:
        df_cwa = cwa_data.get_samples()
    # format cwa times
    df_cwa.time = pd.to_datetime(df_cwa.time)
    
    return df_cwa

def read_sync_file(DIR_DATA, file_sync):
    # read sync file
    df_sync = pd.read_csv(f'{DIR_DATA}/{file_sync}')
    # get sync hour:min:sec
    df_sync['hms'] = [time.split(' ')[4] for time in df_sync.time.values]
    # format into datetime
    df_sync['hms'] = pd.to_datetime(df_sync.hms)
    
    return df_sync

def align_times(df_cwa, df_sync):
    # get year, month, day from cwa
    cwa_Y, cwa_M, cwa_D = df_cwa.time.dt.year.values[0], df_cwa.time.dt.month.values[0], df_cwa.time.dt.day.values[0]
    # match date with cwa
    df_sync['hms'] = df_sync.hms.map(lambda t: t.replace(year=cwa_Y, month=cwa_M, day=cwa_D))
    
    return df_sync

def split_trials(df_cwa, df_sync):
    # drop invalid trials
    df_sync = df_sync.drop(df_sync[df_sync.valid == False].index)
    # split CWA file into trials
    for t, trial in enumerate(np.unique(df_sync.trial.values)):
        # get trial times
        df_sync_t = df_sync[df_sync.trial==trial]
        t_trial = (pd.to_datetime(df_sync_t.loc[df_sync_t.type=='start', 'hms'].values[0]),
                   pd.to_datetime(df_sync_t.loc[df_sync_t.type=='stop', 'hms'].values[0]))
        # mask CWA file based on trial times
        mask = (df_cwa.time >= t_trial[0]) & (df_cwa.time < t_trial[1])
        df_cwa_t = df_cwa.loc[mask].copy()
        df_cwa_t.insert(0, 'trial', trial)
        
        # export trial csv file
        out_filename = f'{Path(file_cwa).stem}_trial_{str(trial).zfill(3)}.csv'
        df_cwa_t.to_csv(f'{DIR_DATA}/{out_filename}', index=False)
        print(f'Trial {trial} exported to: {DIR_DATA}/{out_filename}')
        
if __name__ == "__main__":
    DIR_DATA = sys.argv[1]
    file_cwa = sys.argv[2]
    file_sync = sys.argv[3]
    
    df_cwa = read_axivity(DIR_DATA, file_cwa)
    df_sync = read_sync_file(DIR_DATA, file_sync)
    df_sync = align_times(df_cwa, df_sync)
    split_trials(df_cwa, df_sync)
