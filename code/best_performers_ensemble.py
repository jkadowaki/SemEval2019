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
from PIL import Image


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

def pairwise_f1(probability_matrix, gold_labels, threshold=0.45):
    """
    Ensemble the nearest num predictions.
    """
    
    num_epochs, num_examples = np.shape(probability_matrix)
    ensemble_f1 = -np.ones([num_epochs, num_epochs])
    
    for idx1, vec1 in enumerate(probability_matrix):
        for idx2, vec2 in enumerate(probability_matrix):
            
            if idx1 < idx2:
                continue
            
            prediction = ((vec1 + vec2)/2 > threshold).astype(int)
            try:
                f1 = em.f1_score(prediction, gold_labels[0], average='macro')
            except:
                f1 = em.f1_score(prediction, gold_labels[0,1:], average='macro')
            
            ensemble_f1[idx1, idx2] = f1
            ensemble_f1[idx2, idx1] = f1
    
    return ensemble_f1


################################################################################


def main(verbose=False):
    
    # Directories
    data_dir  = 'eval_data'
    plots_dir = 'plots'

    # Data File
    eval_metrics_file = 'eval_metrics.csv'
    eval_metrics_pickle_file = 'eval_metrics.pkl'

    # Generated Files
    plot_ol_pairf1 = "overlap_pairf1.pdf"

    image_overlap = "overlap_matrix_fold{0}.pdf"
    image_cspred  = "cspred_matrix_fold{0}.pdf"
    image_csprob  = "csprob_matrix_fold{0}.pdf"
    image_pairf1  = "pairf1_fold{0}.pdf"
    
    # Constants
    max_epoch = 100
    
    # Load Data
    try:
        eval = pd.read_pickle(os.path.join(data_dir, eval_metrics_pickle_file))
    except:
        eval = pd.read_csv(os.path.join(data_dir, eval_metrics_file), sep='\t')
        convert_str_to_array(eval, ['accuracy', 'f1_macro', 'f1_micro', 'gold_label',
                                    'f1_weighted', 'probability', 'prediction'])
        eval.to_pickle(os.path.join(data_dir, eval_metrics_pickle_file))


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

    #for ax in [ax7,ax8,ax9,ax10,ax11,ax12]:
    #    ax.plot([0,10], [0,0], 'k-', linewidth=1)


    folds = np.unique(eval['fold'])
    

    for f in folds:

        df_fold     = eval[eval['fold']==f]
        epochs      = df_fold['epoch'].values
        probability = np.vstack(df_fold['probability'].values)
        predictions = np.vstack(df_fold['prediction'].values)
        gold_labels = np.vstack(df_fold['gold_label'].values)
        f1_scores   = df_fold['f1_macro'].values[0]


        # Similarity Metrics  (Matrix of Size [#epochs x #epochs])
        cos_sim_pred = sm.cosine_similarity(predictions)
        cos_sim_prob = sm.cosine_similarity(probability)
        overlap_pred = sm.overlap_percentage(predictions)

        # PLOT: Pairwise Overlap
        plt.figure()
        plt.imshow(100*overlap_pred, interpolation='nearest')
        plt.clim(50,100)
        plt.colorbar()
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


        pairf1 = pairwise_f1(probability, gold_labels, threshold=0.45)
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
    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(2)
    ax1.set_xlim(0,100)
    ax1.set_ylim(0.6, 0.85)
    ax1.set_xlabel(r"$\mathrm{Overlap \, (\%)}$")
    ax1.set_ylabel(r"$\mathrm{Macro-F1 \, Score}$")
    ax1.set_title(r"$n=2 \mathrm{\, Ensemble \, Predictions}$")
    fig1.savefig(os.path.join(plots_dir, plot_ol_pairf1),
                 bbox_inches='tight')
    fig1.clf()


    
################################################################################

if __name__ == '__main__':
    main()


