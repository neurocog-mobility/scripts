"""
@author: Ahomagai

"""

import numpy as np
from numpy.fft import fft, rfft

class FreezeIndexCalculator:
    """ Calculates freeze index, freezeband, motorband, and total power of 
        a given time-series signal. FI is calculated using a hanning window convolution based on 
        methods of Cockx et al. (2023) Journal of NeuroEngineering and Rehabilitation. 
        
        Parameters
        ----------
        data : array_like
            The input data as a one-dimensional array.
        fs: int
            Sampling frequency of the data.
        window: int
            Window length in seconds.

        Returns
        -------

        freeze_index, freeze_band, motor_band, total_power: array_like
            one-dimensional arrays, and can be accessed using the properties freeze_index,
            freeze_band, motor_band, and total_power as dot notation on the class object.
            
    """
    def __init__(self, data, fs, window):
        self.data = data
        self.fs = fs
        self.window_size = int(window * fs)
        self.total_length = len(data)
        
        self.han_win = np.hanning(self.window_size)
        
        self.FI = np.zeros(self.total_length)
        self.freezeband = np.zeros(self.total_length)
        self.motorband = np.zeros(self.total_length)
        self.TPower = np.zeros(self.total_length)
        
        self.calculate_freeze_index()

    def calculate_freeze_index(self):
        for i in range(self.total_length):
            if i <= self.window_size / 2 or i > self.total_length - self.window_size / 2 + 1:
                continue # iterate over the data so you find the window size to calculate FI
            else:
                data_win = self.data[int(i - self.window_size / 2):int(i + self.window_size / 2 - 1)]  
                data_han = self.han_win[0:-1] * data_win  # apply hanning taper
                pow_spctr = fft(data_han)  
                power = np.abs(pow_spctr) ** 2 / self.window_size  
                f = np.arange(0, (self.window_size - 1)) * (self.fs / self.window_size) 
            
                # auc of freezing and motor band
                foi_freezing = np.where((f > 3) & (f <= 8))[0]  # freezing band = 3-8 Hz
                foi_motor = np.where((f >= 0.5) & (f <= 3))[0]  # motor band = 0.5-3 Hz
                auc_freezing = np.trapz(power[foi_freezing]) 
                auc_motor = np.trapz(power[foi_motor])
                auc_total = np.trapz(power[np.concatenate((foi_motor, foi_freezing))])
            
                # calculate the freezing index
                FI_chl = auc_freezing ** 2 / auc_motor ** 2  # squared auc freezing band/squared auc motor band
                FI_norm = np.log(FI_chl * 100)  # normalize
            
                # store variables
                self.FI[i] = FI_norm
                self.freezeband[i] = np.log(auc_freezing * 100)
                self.motorband[i] = np.log(auc_motor * 100)
                self.TPower[i] = np.log(auc_total * 100)
    
    @property
    def freeze_index(self):
        return self.FI
    
    @property
    def freeze_band(self):
        return self.freezeband
    
    @property
    def motor_band(self):
        return self.motorband
    
    @property
    def total_power(self):
        return self.TPower  