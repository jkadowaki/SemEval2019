#!/usr/bin/env python

from __future__ import print_function
from collections import Counter
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
from load_data import load_data, extract_fold_metrics

from sklearn.metrics import accuracy_score, f1_score

################################################################################

def best_performers(df, min_occurence=3, num_ensemble=5, n_best=12,
                    fold=None, plot_file=None):

    """
    
    Args:
    min_occurence (int):
    num_ensembles (int):
    n_best (int):
    fold (int):
    plot_file (str):
    
    Returns:
    epochs_list (list):
    """

    # List of All Folds in Data Frame
    fold_list = list(np.unique(df['fold']))
    
    # Iterates through All but Specified Fold
    if fold:
        fold_list.remove(fold)

    # Top Performing Epochs to Keep
    top_epochs_all = []
    

    for f in fold_list:
        
        # Fold Metrics
        epochs, f1_scores, gold_labels, \
            predictions, probability = extract_fold_metrics(df, fold=f)
    
        top_epochs = sorted( range(len(f1_scores)),
                             reverse=True,
                             key=lambda k: f1_scores[k] )

        top_epochs_all.extend(top_epochs[:n_best])


    # Counts Frequency Each Index
    epoch_counts = Counter(top_epochs_all)
    labels, values = zip(*epoch_counts.items())
    
    # Removes Epochs Below min_occurence Threshold
    best_epochs = [l for idx,l in enumerate(labels) if values[idx] >= min_occurence]
    frequency   = [v for v in values if v >= min_occurence]
    
    # Reorder by Epoch Number
    frequency   = [x for _,x in sorted(zip(best_epochs,frequency))]
    best_epochs = sorted(best_epochs)


    if plot_file:
        x = np.arange(len(best_epochs))
        
        plt.figure()
        plt.bar(x, frequency, 0.95)
        plt.xticks(x, best_epochs)
        plt.xlabel(r"$\mathrm{Epochs}$")
        plt.ylabel(r"$\mathrm{Frequency}$")
        plt.title(r"$\mathrm{Best \, Performing \, Epochs}$")
        plt.savefig(plot_file)
        plt.close()

    return best_epochs[:num_ensemble]


################################################################################

def main(verbose=False):

    ##### Directories #####
    data_dir  = 'eval_data'
    plots_dir = 'plots'

    ###### Data File ######
    eval_metrics_pickle_file = 'eval_metrics.pkl'

    ### Generated Files ###
    plot_freq_idx = "best_epochs{0}.pdf"

    ###### Constants ######
    max_epoch = 100


    # LOAD DATA
    # Columns: Index(['accuracy', 'epoch', 'f1_macro', 'f1_micro', 'f1_weighted',
    #                 'fold', 'gold_label', 'prediction', 'probability']
    df_eval = load_data(data_dir, eval_metrics_pickle_file)


    ############################################################################
    #                        Ensemble: Best Performers                         #
    ############################################################################

    folds = list(np.unique(df_eval['fold']))

    for f in folds:
        
        # Best Epochs for Each Fold
        best = best_performers(df_eval, min_occurence=3, num_ensemble=3,
                               n_best=12, fold=f,
                   plot_file=os.path.join(plots_dir, plot_freq_idx.format(f)) )
        print("Fold {0}:".format(f), "\tEpochs:", best)


    ############################################################################
    #                       Ensemble of Epochs {2,3,4}                         #
    #            Note: Integrate this code with the for loop above.            #
    ############################################################################

    df_best = df_eval[df_eval['epoch'].isin(best)]
    
    for f in folds:
        predictions = np.vstack(df_best[df_best['fold'] == f]["prediction"].values)
        ens_pred = (np.sum(predictions, axis=0)/np.size(best) > 0.5).astype(int)
        gold = df_best[df_best["fold"]==f]['gold_label'].values[0]
        
        if np.size(ens_pred) < np.size(gold):
            gold = gold[1:]
        
        f1_pred = f1_score(ens_pred, gold, average='macro')

        probability = np.vstack(df_best[df_best['fold'] == f]["probability"].values)
        ens_prob = (np.sum(probability, axis=0)/np.size(best) > 0.4).astype(int)
        f1_prob = f1_score(ens_prob, gold, average='macro')

        print("Fold {0}".format(f),
              "Macro-F1 (Pred):", f1_pred,
              "Macro-F1 (Prob):", f1_prob)


    best = best_performers(df_eval, min_occurence=3, num_ensemble=5, n_best=12,
               plot_file=os.path.join(plots_dir, plot_freq_idx.format("_all")) )



################################################################################

if __name__ == '__main__':
    main()


