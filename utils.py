#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:29:26 2024

@author: abdulzaf
"""
import datetime
import numpy as np
from nimbalwear.data import Device

def moving_window_average(signal, window_size):
    ret = np.cumsum(signal, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:]  = ret[window_size - 1:] / window_size
    
    return ret

def trim_data(device: Device, lbl: str, ts: list, te: list):
    """

    Parameters
    ----------
    device : nimbal toolkit Device object
    lbl : Device signal header
    ts : start date-time list
        [year, month, day, hour, minute, second]
    te : end date-time list
        [year, month, day, hour, minute, second]

    Returns
    -------
    time: array of trimmed time values
    raw: array of trimmed raw data

    """
    
    raw = device.signals[device.get_signal_index(lbl)]
    time = device.get_timestamps(lbl, 'datetime')
    
    dt_ts = datetime.datetime(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    dt_te = datetime.datetime(te[0], te[1], te[2], te[3], te[4], te[5])
    
    mask = np.logical_and(time >= dt_ts, time < dt_te)
    
    return time[mask], raw[mask]