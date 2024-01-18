#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:19:23 2024

@author: abdulzaf
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
# SAMPLE ENTROPY

def get_number_of_matches(x, threshold, metric):
    """
    Find number of matches in a set of templates

    Parameters
    ----------
    x : array
        dimensions nxm: n is # of templates, m is embedding dimension
    threshold : distance threshold
    metric : distance metric
        default "chebyshev"; can also use "euclidean"

    Returns
    -------
    float
        number of template matches

    """
    # get distances
    distances = squareform(pdist(x, metric))
    # threshold distances according to tolerance
    distances[distances < threshold] = 0
    distances[distances > 0] = 1
    # invert binary matrix to find matches
    distances = 1 - distances
    # extract number of matches
    distances_upper = distances[np.triu_indices(len(distances))]
    num_matches = np.sum(distances_upper)
    
    return num_matches

def entropy_sample(time_series, m=2, rad="std", tol=1, metric="chebyshev"):
    """
    Computes Sample Entropy

    Parameters
    ----------
    time_series : array
        time series to compute entropy of
    m : embedding dimension
        size of shorter template; longer template taken as m+1
    rad : radius
        distance threshold radius; default "std" takes time series standard
        deviation as the radius, otherwise takes a float value
    tol : tolerance
        percentage of radius to take as a distance threshold
    metric : distance metric
        default "chebyshev"; can also use "euclidean"

    Returns
    -------
    float
        entropy of time series or -1 if intractable (no matches found)

    """
    # convert time_series to array
    time_series = np.array(time_series)
    # set standard deviation as base radius if none provided
    if rad=="std":
        threshold = tol * np.std(time_series)
    else:
        threshold = tol * rad
    # get list of templates for dimensions m, m_1
    templates_1 = np.array([time_series[i:-1*(m-i+1)] for i in range(m)]).T
    templates_2 = np.array([time_series[i:-1*(m-i+1)] for i in range(m+1)]).T
    # get number of matches at each template dimension
    A = get_number_of_matches(templates_2, threshold, metric)
    B = get_number_of_matches(templates_1, threshold, metric)
    # compute sample entropy
    if (A > 0) and (B > 0):
        samp_entropy = -1*np.log(A/B)
    else:
        samp_entropy = -1
    
    return samp_entropy
