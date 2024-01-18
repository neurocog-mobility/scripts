#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:17:23 2024

@author: abdulzaf
"""
import numpy as np
from collections import Counter

# CONDITIONAL ENTROPY


def entropy_conditional(seq):
    """
    Computes Conditional Entropy

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
        # get unique symbols in sequence
        states = np.unique(seq)
        # create transition matrix
        tmat = np.zeros((len(states), len(states)))
        for i in range(len(seq)-1):
            i1 = seq[i]
            i2 = seq[i+1]
            tmat[states == i1, states == i2] += 1

        # normalize transition probabilities
        tmat = tmat/np.sum(tmat)
        # find probability of x occurring
        prob_x = np.sum(tmat, axis=1)
        # find H(y|X=x): the Shannon entropy of y, given x
        prob_y = [abs(np.nansum((tmat[row, :]/sum(tmat[row, :])) *
                  np.log2((tmat[row, :]/sum(tmat[row, :]))))) for row in range(len(tmat))]

        if np.isnan(abs(sum(prob_x*prob_y))):
            return -1
        else:
            return abs(sum(prob_x*prob_y))
    else:
        return -1
