# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:19:44 2021

@author: zafar
"""
import numpy as np
from collections import Counter

# STATIONARY ENTROPY


def entropy_stationary(seq):
    """
    Computes Stationary Entropy (Shannon Entropy)

    Parameters
    ----------
    seq : list
        sequence of symbols to compute entropy of

    Returns
    -------
    float
        entropy of sequence or -1 if sequence is empty

    """
    if len(seq) > 0:
        # get number of occurrences for each unique symbol in sequence
        counts = Counter(seq)
        # get each unique symbol in sequence
        states = np.array(list(counts.keys()))
        # get (non-normal) probability distribution for symbols
        state_counts = np.array([counts[i] for i in states])
        # get normalized probability distribution for symbols
        state_prob = state_counts/np.sum(state_counts)
        # return entropy value
        if np.isnan(abs(sum(state_prob * np.log2(state_prob)))):
            return -1
        else:
            return abs(sum(state_prob * np.log2(state_prob)))
    else:
        return -1
