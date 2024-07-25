#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:38:06 2024

@author: abdulzaf
"""
import numpy as np
from scipy import signal
from utils import moving_window_average
def compute_hrv(rr, fs, method='rmssd'):
    """

    """
    if method=='rmssd':
        rr_interval = np.diff(np.where(rr==1)[0]) / fs
        
    rmssd = np.sqrt(np.mean(np.diff(rr_interval)**2))
    
    return rmssd

def pan_tompkins_detector(fs, ecg):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm. 
    In: IEEE Transactions on Biomedical Engineering 
    BME-32.3 (1985), pp. 230–236.
    """
    
    maxQRSduration = 0.150 #sec
    f1 = 5/fs
    f2 = 15/fs

    b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, ecg)        

    diff = np.diff(filtered_ecg) 

    squared = diff*diff

    N = int(maxQRSduration*fs)
    mwa = moving_window_average(squared, N)
    mwa[:int(maxQRSduration*fs*2)] = 0

    pks, _ = signal.find_peaks(mwa, distance = 0.3*fs, height=np.mean(mwa))

    return mwa, pks

def ecg2rr(fs, ecg):
    mwa, pks = pan_tompkins_detector(fs, ecg)
    
    rr = np.zeros(len(mwa))
    rr[pks] = 1
    
    return rr