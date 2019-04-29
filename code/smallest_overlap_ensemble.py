#!/usr/bin/env python

from __future__ import print_function
import csv
import errno
import glob
from itertools import combinations
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
from similarity_metrics import overlap_percentage, average_pairwise
import evaluation_metrics as em
import evaluate_results   as er
import similarity_metrics as sm
from load_data  import load_data, extract_fold_metrics

from sklearn.metrics import accuracy_score, f1_score


################################################################################


def main(verbose=False):
    
    ##### Directories #####
    data_dir  = 'eval_data'
    plots_dir = 'plots'

    ###### Data File ######
    eval_metrics_file = 'eval_metrics.csv'
    eval_metrics_pickle_file = 'eval_metrics.pkl'

    ### Generated Files ###
    plot_ol_pairf1 = "overlap_pairf1.pdf"

    image_overlap = "overlap_matrix_fold{0}.pdf"
    image_cspred  = "cspred_matrix_fold{0}.pdf"
    image_csprob  = "csprob_matrix_fold{0}.pdf"
    image_pairf1  = "pairf1_fold{0}.pdf"
    
    ###### Constants ######
    max_epoch = 100
    
    
    # LOAD DATA
    # Columns: Index(['accuracy', 'epoch', 'f1_macro', 'f1_micro', 'f1_weighted',
    #                 'fold', 'gold_label', 'prediction', 'probability']
    df_eval = load_data(data_dir, eval_metrics_pickle_file)


    ############################################################################
    #                    Ensemble: Smallest Overlap (n=2,3,5)                  #
    ############################################################################
    
    fig1, ax1 = plt.subplots()  # F1-Macro for Ensemble E2 vs. Overlap


    # Plot Baselines
    xlim=[0,100]
    for ax in [ax1]:
        ax.plot(xlim, [0.829]*2, 'r--', label=r"$\mathrm{Top \, System}$")
        ax.plot(xlim, [0.800]*2, 'g--', label=r"$\mathrm{CNN}$")
        ax.plot(xlim, [0.750]*2, 'k--', label=r"$\mathrm{BiLSTM}$")
        ax.plot(xlim, [0.690]*2, 'b--', label=r"$\mathrm{SVM}$")


    folds = np.unique(df_eval['fold'])
    

    for f in folds:

        # Fold Metrics
        epochs, f1_scores, gold_labels, \
            predictions, probability = extract_fold_metrics(df_eval, fold=f)
                

        # Similarity Metrics  (Matrix of Size [#epochs x #epochs])
        cos_sim_pred = sm.cosine_similarity(predictions)
        cos_sim_prob = sm.cosine_similarity(probability)
        overlap_pred = sm.overlap_percentage(predictions)

        # PLOT: Pairwise Overlap
        plt.figure()
        plt.imshow(100*overlap_pred, interpolation='nearest')
        plt.clim(50,100)
        plt.colorbar()
        plt.xlabel(r"$\mathrm{Epoch}$")
        plt.ylabel(r"$\mathrm{Epoch}$")
        plt.title(r"$\mathrm{Pairwise \, Overlap \, (\%)}$")
        plt.savefig(os.path.join(plots_dir, image_overlap.format(f)),
                    bbox_inches='tight')
        plt.close()
        
        # PLOT: Pairwise Cosine Similarity for Predictions
        plt.figure()
        plt.imshow(cos_sim_pred, interpolation='nearest')
        plt.clim(0.5,1)
        plt.colorbar()
        plt.savefig(os.path.join(plots_dir, image_cspred.format(f)),
                    bbox_inches='tight')
        plt.close()
        
        # PLOT: Pairwise Cosine Similarity for Probabilities
        plt.figure()
        plt.imshow(cos_sim_prob, interpolation='nearest')
        plt.clim(0.5,1)
        plt.colorbar()
        plt.savefig(os.path.join(plots_dir, image_csprob.format(f)),
                    bbox_inches='tight')
        plt.close()


        pairf1 = em.pairwise_f1(probability, gold_labels, threshold=0.45)
        plt.figure()
        plt.imshow(pairf1, interpolation='nearest')
        plt.clim(0.6,0.85)
        plt.colorbar()
        plt.savefig(os.path.join(plots_dir, image_pairf1.format(f)),
                    bbox_inches='tight')
        plt.close()


        ax1.scatter(100*overlap_pred, pairf1, marker='.', s=0.1,
                    label=r"$\mathrm{{Fold \, {0} }}$".format(f))

    legend=ax1.legend(loc='lower center', prop={'size': 10}, ncol=4, frameon=True)
    for legend_handle in legend.legendHandles[4:]:
        legend_handle._sizes = [10]
    ax1.set_xlim(0,100)
    ax1.set_ylim(0.6, 0.85)
    ax1.set_xlabel(r"$\mathrm{Overlap \, (\%)}$")
    ax1.set_ylabel(r"$\mathrm{Macro-F1 \, Score}$")
    ax1.set_title(r"$n=2 \mathrm{\, Ensemble \, Scores}$")
    fig1.savefig(os.path.join(plots_dir, plot_ol_pairf1),
                 bbox_inches='tight')
    fig1.clf()
    

    # Super crap code... extremely inefficient. rewrite later!
    # tl;dr:

    epochs, f1_scores, gold_labels, \
        predictions, probability = extract_fold_metrics(df_eval, fold=9)
    epoch_combo = [(e1, e2, e3) for e1, e2, e3 in combinations(range(epochs.size), r=3)]
    overlap = overlap_percentage(predictions)
    f1 = []
    avg_overlap = []

    for idx, ec in enumerate(epoch_combo):
        
        if idx>35000==0:
            break
        try:
            pred = em.get_ensemble_prediction(df_eval, 9, ec)
            
            f1.append(f1_score(pred, gold_labels[0], average='macro'))
            avg_overlap.append(sm.average_pairwise(overlap, ec))
        except:
            pass

    plt.figure()
    plt.plot(xlim, [0.829]*2, 'r--', label=r"$\mathrm{Top \, System}$")
    plt.plot(xlim, [0.800]*2, 'g--', label=r"$\mathrm{CNN}$")
    plt.plot(xlim, [0.750]*2, 'k--', label=r"$\mathrm{BiLSTM}$")
    plt.plot(xlim, [0.690]*2, 'b--', label=r"$\mathrm{SVM}$")

    plt.scatter(avg_overlap, f1,  s=0.1, marker='.')
    plt.xlabel(r"$\mathrm{Average \, Pairwise \, Overlap \, (\%)}$")
    plt.ylabel(r"$\mathrm{Macro-F1 \, Score}$")
    plt.ylim(.60,.85)
    plt.title(r"$n=3 \mathrm{\, Ensemble \, Scores}$")
    plt.legend(loc='lower center', prop={'size': 10}, ncol=4, frameon=True)
    plt.savefig('n3_f1_overlap9.pdf')
    plt.close()



################################################################################

if __name__ == '__main__':
    main()


