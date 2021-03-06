#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


################################################################################

def get_prediction_matrix(df, fold, epochs=None, prob=False):
    
    """
        Retrieves the predictions given the epochs and fold.
        
        Args:
        df (pd.DataFrame):
        fold (int):
        epochs (iterable):
        prob (boolean):
        
        Returns:
        Prediction or probability matrix for the specified epochs in a fold
        """
    
    if epochs:
        df_epochs = df[df['epoch'].isin(epochs)]
    else:
        df_epochs = df
    
    if prob:
        return np.vstack(df_epochs[df_epochs['fold'] == fold]["probability"].values)
    
    return np.vstack(df_epochs[df_epochs['fold'] == fold]["prediction"].values)


################################################################################

def get_ensemble_prediction(df, fold, epochs, threshold=0.5, prob=False):
    
    """
    Computes the predictions given the epochs to ensemble.
    
    Args:
    df (pd.DataFrame):
    fold (int):
    epochs (iterable):
    threshold (float):
    prob (boolean):
    
    Returns:
    Ensemble prediction or probability.
    """
    
    prediction = get_prediction_matrix(df, fold, epochs, prob=prob)
    
    return (np.sum(prediction, axis=0)/np.size(epochs) > threshold).astype(int)


################################################################################

def accuracy(prediction_matrix, gold_matrix):
    
    """
    Computes the accuracy between all corresponding predictions and gold labels.
    
    Args:
    prediction_matrix (np.ndarray): 2D-array of size (#epochs x #predictions)
    
    Returns:
    (np.ndarray): 1D-array of size (#epochs)
    """

    try:
        return np.array([ accuracy_score(row, gold_matrix[idx])     \
                          for idx,row in enumerate(prediction_matrix) ])
    
    except:
        raise ValueError( "Dimensions {0} != {1}".format(
                           prediction_matrix.shape, gold_matrix.shape) )



################################################################################

def f1_metric(prediction_matrix, gold_matrix, average='macro'):
    
    """
    Computes the accuracy between ___ and ___.
    
    Args:
    prediction_matrix (np.ndarray): 2D-array of size (#epochs x #predictions)
    average(str): 'micro', 'macro', 'weighted'
    
    Returns:
    f1 (np.ndarray): 2D-array of size (#epochs x #epochs)
    """
    
    try:
        return np.array([ f1_score(row, gold_matrix[idx], average=average) \
                          for idx,row in enumerate(prediction_matrix) ])
     
    except:
        raise ValueError( "Dimensions {0} != {1}".format(
                           prediction_matrix.shape, gold_matrix.shape) )


################################################################################

def pairwise_f1(probability_matrix, gold_labels, threshold=0.45):
    
    """
    Ensemble the nearest num predictions.
    
    Args:
    probability_matrix (np.ndarray):
    gold_labels (np.ndarray):
    threshold (float):
    
    Returns:
    ensemble_f1 (np.ndarray):
    """
    
    num_epochs, num_examples = np.shape(probability_matrix)
    ensemble_f1 = -np.ones([num_epochs, num_epochs])
    
    for idx1, vec1 in enumerate(probability_matrix):
        for idx2, vec2 in enumerate(probability_matrix):
            
            if idx1 < idx2:
                continue
            
            prediction = ((vec1 + vec2)/2 > threshold).astype(int)
            try:
                f1 = f1_score(prediction, gold_labels[0], average='macro')
            except:
                f1 = f1_score(prediction, gold_labels[0,1:], average='macro')
            
            ensemble_f1[idx1, idx2] = f1
            ensemble_f1[idx2, idx1] = f1
    
    return ensemble_f1

################################################################################
