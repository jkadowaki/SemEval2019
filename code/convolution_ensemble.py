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
from load_data  import load_data, extract_fold_metrics


################################################################################

def sliding_window(arr, num=3, threshold=0.5):
    """
    Ensemble the nearest num predictions.
    """
    
    num_row, num_col    = arr.shape
    ensemble_prediction = np.empty([num_row - num + 1, num_col])
    """
    for idx in range(num):
        if idx < num-1:
            ensemble_prediction += arr[idx:-num+1+idx]
        else:
            ensemble_prediction += arr[idx:]
            
    return (ensemble_prediction/num > threshold).astype(int)
    """
    
    for idx in range(ensemble_prediction.shape[0]):
        ensemble_prediction[idx,:] = (np.sum(arr[idx:num+idx,:], axis=0)/num > threshold).astype(int)
    
    return ensemble_prediction


################################################################################


def main(verbose=False):
    
    ##### Directories #####
    data_dir  = 'eval_data'
    plots_dir = 'plots'

    ###### Data File ######
    eval_metrics_file = 'eval_metrics.csv'
    eval_metrics_pickle_file = 'eval_metrics.pkl'
    
    ### Generated Files ###
    plot_f1_ens2sw_pred = "f1_e2sw_pred.pdf"
    plot_f1_ens3sw_pred = "f1_e3sw_pred.pdf"
    plot_f1_ens5sw_pred = "f1_e5sw_pred.pdf"
    plot_f1_ens2sw_prob = "f1_e2sw_prob.pdf"
    plot_f1_ens3sw_prob = "f1_e3sw_prob.pdf"
    plot_f1_ens5sw_prob = "f1_e5sw_prob.pdf"

    plot_deltaf1_ens2sw_pred = "deltaf1_e2sw_pred.pdf"
    plot_deltaf1_ens3sw_pred = "deltaf1_e3sw_pred.pdf"
    plot_deltaf1_ens5sw_pred = "deltaf1_e5sw_pred.pdf"
    plot_deltaf1_ens2sw_prob = "deltaf1_e2sw_prob.pdf"
    plot_deltaf1_ens3sw_prob = "deltaf1_e3sw_prob.pdf"
    plot_deltaf1_ens5sw_prob = "deltaf1_e5sw_prob.pdf"

    ###### Constants ######
    max_epoch = 100
    
    
    # LOAD DATA
    # Columns: Index(['accuracy', 'epoch', 'f1_macro', 'f1_micro', 'f1_weighted',
    #                 'fold', 'gold_label', 'prediction', 'probability']
    df_eval = load_data(data_dir, eval_metrics_pickle_file)


    ############################################################################
    #                     Ensemble: Sliding Window (n=2,3,5)                   #
    ############################################################################

    fig1, ax1 = plt.subplots()  # F1-Macro for Ensemble E2SW_Pred vs. Epoch
    fig2, ax2 = plt.subplots()  # F1-Macro for Ensemble E3SW_Pred vs. Epoch
    fig3, ax3 = plt.subplots()  # F1-Macro for Ensemble E5SW_Pred vs. Epoch
    fig4, ax4 = plt.subplots()  # F1-Macro for Ensemble E2SW_Prob vs. Epoch
    fig5, ax5 = plt.subplots()  # F1-Macro for Ensemble E3SW_Prob vs. Epoch
    fig6, ax6 = plt.subplots()  # F1-Macro for Ensemble E5SW_Prob vs. Epoch

    fig7,  ax7  = plt.subplots()  # Delta F1-Macro for Ensemble E2SW_Pred vs. Epoch
    fig8,  ax8  = plt.subplots()  # Delta F1-Macro for Ensemble E3SW_Pred vs. Epoch
    fig9,  ax9  = plt.subplots()  # Delta F1-Macro for Ensemble E5SW_Pred vs. Epoch
    fig10, ax10 = plt.subplots()  # Delta F1-Macro for Ensemble E2SW_Prob vs. Epoch
    fig11, ax11 = plt.subplots()  # Delta F1-Macro for Ensemble E3SW_Prob vs. Epoch
    fig12, ax12 = plt.subplots()  # Delta F1-Macro for Ensemble E5SW_Prob vs. Epoch

    # Plot Baselines
    xlim=[0,100]
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.plot(xlim, [0.829]*2, 'r--', label=r"$\mathrm{Top \, System}$")
        ax.plot(xlim, [0.800]*2, 'g--', label=r"$\mathrm{CNN}$")
        ax.plot(xlim, [0.750]*2, 'k--', label=r"$\mathrm{BiLSTM}$")
        ax.plot(xlim, [0.690]*2, 'b--', label=r"$\mathrm{SVM}$")

    for ax in [ax7,ax8,ax9,ax10,ax11,ax12]:
        ax.plot([0,10], [0,0], 'k-', linewidth=1)

    folds = np.unique(df_eval['fold'])

    for f in folds:
        
        # Fold Metrics
        epochs, f1_scores, gold_labels, \
            predictions, probability = extract_fold_metrics(df_eval, fold=f)
        

        # Sliding Window Predictions
        ens2sw_pred = sliding_window(predictions, 2)
        ens3sw_pred = sliding_window(predictions, 3)
        ens5sw_pred = sliding_window(predictions, 5)
        ens2sw_prob = sliding_window(probability, 2, threshold=0.45)
        ens3sw_prob = sliding_window(probability, 3, threshold=0.45)
        ens5sw_prob = sliding_window(probability, 5, threshold=0.45)

        # Macro-F1 Scores of Sliding Window Ensemble
        f1_ens2sw_pred = em.f1_metric(ens2sw_pred, gold_labels)
        f1_ens3sw_pred = em.f1_metric(ens3sw_pred, gold_labels)
        f1_ens5sw_pred = em.f1_metric(ens5sw_pred, gold_labels)
        f1_ens2sw_prob = em.f1_metric(ens2sw_prob, gold_labels)
        f1_ens3sw_prob = em.f1_metric(ens3sw_prob, gold_labels)
        f1_ens5sw_prob = em.f1_metric(ens5sw_prob, gold_labels)

        # Delta Macro-F1 Scores of Sliding Window Ensemble
        deltaf1_ens2sw_pred = f1_ens2sw_pred - f1_scores[1:f1_ens2sw_pred.size+1]
        deltaf1_ens3sw_pred = f1_ens3sw_pred - f1_scores[1:f1_ens3sw_pred.size+1]
        deltaf1_ens5sw_pred = f1_ens5sw_pred - f1_scores[2:f1_ens5sw_pred.size+2]
        deltaf1_ens2sw_prob = f1_ens2sw_prob - f1_scores[1:f1_ens2sw_prob.size+1]
        deltaf1_ens3sw_prob = f1_ens3sw_prob - f1_scores[1:f1_ens3sw_prob.size+1]
        deltaf1_ens5sw_prob = f1_ens5sw_prob - f1_scores[2:f1_ens5sw_prob.size+2]


        # Plots
        ax1.plot( epochs[1:], f1_ens2sw_pred,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax2.plot( epochs[1:-1], f1_ens3sw_pred,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax3.plot( epochs[2:-2], f1_ens5sw_pred,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax4.plot( epochs[1:], f1_ens2sw_prob,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax5.plot( epochs[1:-1], f1_ens3sw_prob,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax6.plot( epochs[2:-2], f1_ens5sw_prob,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )

        ax7.plot( epochs[1:11], deltaf1_ens2sw_pred[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax8.plot( epochs[1:11], deltaf1_ens3sw_pred[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax9.plot( epochs[2:12], deltaf1_ens5sw_pred[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax10.plot( epochs[1:11], deltaf1_ens2sw_prob[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax11.plot( epochs[1:11], deltaf1_ens3sw_prob[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax12.plot( epochs[2:12], deltaf1_ens5sw_prob[:10],
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )



    ax1.legend(loc='lower center', prop={'size': 10}, ncol=4)
    ax1.set_xlim(0,100)
    ax1.set_ylim(0.6, 0.83)
    ax1.set_xlabel(r"$\mathrm{Epochs}$")
    ax1.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax1.set_title(r"$\mathrm{Sliding \, Window \, (n=2) \, Over \, Prediction}$")
    fig1.savefig(os.path.join(plots_dir, plot_f1_ens2sw_pred), bbox_inches = 'tight')
    fig1.clf()

    ax2.legend(loc='lower center', prop={'size': 10}, ncol=4)
    ax2.set_xlim(0,100)
    ax2.set_ylim(0.6, 0.83)
    ax2.set_xlabel(r"$\mathrm{Epochs}$")
    ax2.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax2.set_title(r"$\mathrm{Sliding \, Window \, (n=3) \, Over \, Prediction}$")
    fig2.savefig(os.path.join(plots_dir, plot_f1_ens3sw_pred), bbox_inches = 'tight')
    fig2.clf()

    ax3.legend(loc='lower center', prop={'size': 10}, ncol=4)
    ax3.set_xlim(0,100)
    ax3.set_ylim(0.6, 0.83)
    ax3.set_xlabel(r"$\mathrm{Epochs}$")
    ax3.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax3.set_title(r"$\mathrm{Sliding \, Window \, (n=5) \, Over \, Prediction}$")
    fig3.savefig(os.path.join(plots_dir, plot_f1_ens5sw_pred), bbox_inches = 'tight')
    fig3.clf()

    ax4.legend(loc='lower center', prop={'size': 10}, ncol=4)
    ax4.set_xlim(0,100)
    ax4.set_ylim(0.6, 0.83)
    ax4.set_xlabel(r"$\mathrm{Epochs}$")
    ax4.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax4.set_title(r"$\mathrm{Sliding \, Window \, (n=2) \, Over \, Probability}$")
    fig4.savefig(os.path.join(plots_dir, plot_f1_ens2sw_prob), bbox_inches = 'tight')
    fig4.clf()

    ax5.legend(loc='lower center', prop={'size': 10}, ncol=4)
    ax5.set_xlim(0,100)
    ax5.set_ylim(0.6, 0.83)
    ax5.set_xlabel(r"$\mathrm{Epochs}$")
    ax5.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax5.set_title(r"$\mathrm{Sliding \, Window \, (n=3) \, Over \, Probability}$")
    fig5.savefig(os.path.join(plots_dir, plot_f1_ens3sw_prob), bbox_inches = 'tight')
    fig5.clf()

    ax6.legend(loc='best', prop={'size': 10}, ncol=4)
    ax6.set_xlim(0,100)
    ax6.set_ylim(0.6, 0.83)
    ax6.set_xlabel(r"$\mathrm{Epochs}$")
    ax6.set_ylabel(r"$\mathrm{Macro-F1}$")
    ax6.set_title(r"$\mathrm{Sliding \, Window \, (n=5) \, Over \, Probability}$")
    fig6.savefig(os.path.join(plots_dir, plot_f1_ens5sw_prob), bbox_inches = 'tight')
    fig6.clf()


    ax7.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax7.set_xlim(1,10)
    #ax7.set_ylim(-0.3, 0.4)
    ax7.set_xlabel(r"$\mathrm{Epochs}$")
    ax7.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax7.set_title(r"$\mathrm{Sliding \, Window \, (n=2) \, Over \, Prediction}$")
    fig7.savefig(os.path.join(plots_dir, plot_deltaf1_ens2sw_pred), bbox_inches = 'tight')
    fig7.clf()
    
    ax8.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax8.set_xlim(1,10)
    #ax8.set_ylim(-0.2, 0.25)
    ax8.set_xlabel(r"$\mathrm{Epochs}$")
    ax8.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax8.set_title(r"$\mathrm{Sliding \, Window \, (n=3) \, Over \, Prediction}$")
    fig8.savefig(os.path.join(plots_dir, plot_deltaf1_ens3sw_pred), bbox_inches = 'tight')
    fig8.clf()
    
    ax9.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax9.set_xlim(2,11)
    #ax9.set_ylim(-0.25, 0.25)
    ax9.set_xlabel(r"$\mathrm{Epochs}$")
    ax9.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax9.set_title(r"$\mathrm{Sliding \, Window \, (n=5) \, Over \, Prediction}$")
    fig9.savefig(os.path.join(plots_dir, plot_deltaf1_ens5sw_pred), bbox_inches = 'tight')
    fig9.clf()

    ax10.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax10.set_xlim(1,10)
    #ax10.set_ylim(-0.2, 0.3)
    ax10.set_xlabel(r"$\mathrm{Epochs}$")
    ax10.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax10.set_title(r"$\mathrm{Sliding \, Window \, (n=2) \, Over \, Probability}$")
    fig10.savefig(os.path.join(plots_dir, plot_deltaf1_ens2sw_prob), bbox_inches = 'tight')
    fig10.clf()

    ax11.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax11.set_xlim(1,10)
    #ax11.set_ylim(-0.2,0.25)
    ax11.set_xlabel(r"$\mathrm{Epochs}$")
    ax11.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax11.set_title(r"$\mathrm{Sliding \, Window \, (n=3) \, Over \, Probability}$")
    fig11.savefig(os.path.join(plots_dir, plot_deltaf1_ens3sw_prob), bbox_inches = 'tight')
    fig11.clf()

    ax12.legend(loc='upper right', prop={'size': 10}, ncol=4)
    ax12.set_xlim(2,11)
    #ax12.set_ylim(-0.15, 0.25)
    ax12.set_xlabel(r"$\mathrm{Epochs}$")
    ax12.set_ylabel(r"$\Delta \left(\mathrm{Macro-F1} \right)$")
    ax12.set_title(r"$\mathrm{Sliding \, Window \, (n=5) \, Over \, Probability}$")
    fig12.savefig(os.path.join(plots_dir, plot_deltaf1_ens5sw_prob), bbox_inches = 'tight')
    fig12.clf()


################################################################################

if __name__ == '__main__':
    main()


