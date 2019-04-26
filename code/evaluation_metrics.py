#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

################################################################################

def accuracy(prediction_matrix, gold_matrix):
    
    """
    Computes the accuracy between all corresponding predictions and gold labels.
    
    Args:
    prediction_matrix (nd.array): 2D-array of size (#epochs x #predictions)
    
    Returns:
    (nd.array): 1D-array of size (#epochs)
    """

    try:
        return np.array([ accuracy_score(row, gold_matrix[idx])     \
                          for idx,row in enumerate(prediction_matrix) ])
    
    except:
        raise ValueError( "Dimensions {0} != {1}".format(
                           prediction_matrix.shape, gold_matrix.shape) )



################################################################################

def f1_metric(prediction_matrix, gold_matrix, average='micro'):
    
    """
    Computes the accuracy between ___ and ___.
    
    Args:
    prediction_matrix (nd.array): 2D-array of size (#epochs x #predictions)
    average(str): 'micro', 'macro', 'weighted'
    
    Returns:
    f1 (nd.array): 2D-array of size (#epochs x #epochs)
    """
    
    try:
        return np.array([ f1_score(row, gold_matrix[idx], average=average) \
                          for idx,row in enumerate(prediction_matrix) ])
     
    except:
        raise ValueError( "Dimensions {0} != {1}".format(
                           prediction_matrix.shape, gold_matrix.shape) )

################################################################################
