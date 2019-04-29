#!/usr/bin/env python

from __future__ import print_function
import csv
import errno
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
plt.rc('text', usetex=True)

import numpy as np
np.set_printoptions(suppress=True)
import os

import sys
sys.path.append("./code")
import similarity_metrics as sm
import evaluation_metrics as em
import evaluate_results   as er


################################################################################

def convert_str_to_array(df, columns):
    """
    :Args:
    df (pd.DataFrame)
    columns (List of Strings)
    """
    for col in columns:
        for idx,elem in enumerate(df[col]):
            try:
                df[col][idx]=np.array(elem.strip("[]").split(", ")).astype(float)
            except:
                pass


################################################################################

def load_data(directory, pickle_file=None, tsv_file=None,
              columns=['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                       'gold_label', 'probability', 'prediction'] ):
    
    if not pickle_file and not tsv_file:
        raise Exception("pickle_file or tsv_file must be specified.")
    
    # Load Data
    try:
        df = pd.read_pickle(os.path.join(directory, pickle_file))
    except:
        df = pd.read_csv(os.path.join(directory, tsv_file), sep='\t')
        convert_str_to_array(df, columns)
        df.to_pickle(os.path.join(directory, pickle_file))

    return df


################################################################################

def extract_fold_metrics(df, fold=0):

        """
        """
        
        df_fold     = df[df['fold']==fold]
        epochs      = np.array(df_fold['epoch'].values)
        probability = np.vstack(df_fold['probability'].values)
        predictions = np.vstack(df_fold['prediction'].values)
        gold_labels = np.vstack(df_fold['gold_label'].values)
        f1_scores   = np.array(df_fold['f1_macro'].values[0])

        return epochs, f1_scores, gold_labels, predictions, probability

################################################################################

if __name__ == '__main__':
    load_data()


