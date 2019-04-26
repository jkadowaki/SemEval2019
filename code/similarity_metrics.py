#!/usr/bin/env python

from __future__ import print_function
from itertools import combinations
import numpy as np
np.set_printoptions(suppress=True)


################################################################################

def cosine_similarity(prediction_matrix):

    """
    Computes the pairwise cosine similarity metric for every pair of epochs.
    
    Args:
    prediction_matrix (nd.array): 2D-array of size (#epochs x #predictions)
    
    Returns:
    cos_sim (nd.array): 2D-array of size (#epochs x #epochs)
    """

    norm = np.sqrt( np.sum(prediction_matrix**2, axis=1) )
    norm_matrix = np.tile(norm, [np.size(norm),1] )

    cos_sim = prediction_matrix.dot(prediction_matrix.T) / norm_matrix / norm_matrix.T

    return cos_sim


################################################################################

def overlap_percentage(prediction_matrix):

    """
    Computes the pairwise overlap percentage metric for every pair of epochs.
    
    Args:
    prediction_matrix (nd.array): 2D-array of size (#epochs x #predictions)
    
    Returns:
    (nd.array): 2D-array of size (#epochs x #epochs)
    """

    num_epochs, num_examples = np.shape(prediction_matrix)

    overlap = np.empty([num_epochs, num_epochs])

    for row_idx, vec in enumerate(prediction_matrix):
        overlap[row_idx, :] = np.sum([prediction_matrix[0]==row for row in prediction_matrix], axis=1)

    return overlap / num_examples

################################################################################

def average_pairwise(pairwise_matrix, epochs):

    """
    Computes the average pairwise metric for a list of epochs.
    
    Args:
    pairwise_matrix (nd.array): 2D-array of size (#epochs x #epochs) containing
                                pairwise similarity or evaluation metrics.
    epochs (nd.array / List): List of epoches to average over all combinations
                              of pairwise metrics.
    
    Returns:
    (float): Averaged result of all pair-combinations of indicies.
    """

    epoch_pairs = combinations(epochs, r=2)
    
    return np.mean([pairwise_matrix[e1, e2] for e1,e2 in epoch_pairs])


################################################################################
