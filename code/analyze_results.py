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
import seaborn as sns
plt.rc('text', usetex=True)
sns.set(font_scale=1.5, rc={'text.usetex' : True}, style="white", color_codes=True)

import numpy as np
np.set_printoptions(suppress=True)
import os

import sys
sys.path.append("./code")
import similarity_metrics as sm
import evaluation_metrics as em


################################################################################

def create_directory(path):
    
    """
    Creates a new directory if one does not exist.
    
    Args:
    path (str): Name of directory
    
    """
    
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


################################################################################

def get_results(directory, extension=".pdf", prefix=None):
    
    """
    Walks through specified directory to find all files with specified extension.
    
    Args:
    directory (str) - Name of Directory to Search
    extension (str) - Extension of File to Search
    prefix (str) - Start of Filename to Search.
    
    Returns:
    An iterator of filenames in directory satisfying prefix/extension.
    """
    
    # Traverses Directory Tree via Depth-First Search
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            # Checks Whether Filename Matches Requested Prefix/Extension
            if all([file.endswith(extension),
                    (prefix is None or file.startswith(prefix)) ]):
                
                # Creates an Iterator of File Names
                yield os.path.join(root, file)


################################################################################

def main():
    
    ############################################################################
    #                            FILENAME CONVENSION                           #
    ############################################################################
    
    # Directories
    data_dir        = 'eval_data'
    results_dir     = 'epoch_results'
    plots_dir       = 'plots'
    
    # Gold Labels
    label_file      = 'label.tsv'

    # Constants
    steps_in_epoch  = 373
    
    # Generated Files
    final_eval_file = 'eval_results.csv'
    final_test_file = 'test_results.csv'
    plot_eval            = "dev_results.pdf"
    plot_accuracy        = "accuracy.pdf"
    plot_f1micro         = "f1_micro.pdf"
    plot_f1macro         = "f1_macro.pdf"
    plot_f1weighted      = "f1_weighted.pdf"
    plot_cospred_cosprob = "cospred_cosprob.pdf"
    plot_cospred_overlap = "cospred_overlap.pdf"
    plot_cosprob_overlap = "cosprob_overlap.pdf"
    
    ############################################################################
    #         AGGREGATE ALL INDIVIDUAL RESULTS INTO A SINGLE DATAFRAME         #
    ############################################################################

    # Crawl the Data Directory for Eval & Test Results Files
    eval_files = get_results(data_dir, prefix='eval_results_', extension='.txt')
    test_files = get_results(data_dir, prefix='test_results_', extension='.tsv')
    dev_files  = get_results(data_dir, prefix='dev', extension='.txt')

    # Temporarily Data Storage for All Results Files
    eval_dict_list = []
    test_dict_list = []


    for eval_file in eval_files:
        # Save Evaluation Metrics into Dictionary
        eval_dict = {key:value for key,value in
                    zip(["eval_accuracy", "eval_loss", "global_step", "loss"],
                        np.genfromtxt(eval_file, unpack=True)[2])}
    
        # Compute the Epoch Number
        eval_dict["epoch"] = int(eval_dict.get("global_step") / steps_in_epoch)
        
        # Get the Fold from the File Path
        eval_dict["fold"]  = int(os.path.dirname(eval_file).split('/')[1][4:])
        eval_dict_list.append(eval_dict)


    for test_file in test_files:
        # Get the Fold & Epoch Number from the File Path
        fold  = int(os.path.dirname(test_file).split('/')[1][4:])
        epoch = int( os.path.splitext(os.path.basename(
                       test_file))[0].split('_')[2] ) // steps_in_epoch
        
        # Get Prediction
        try:
            results = np.genfromtxt(test_file, unpack=True, encoding='bytes')[1]
            test_dict_list.append({"fold": fold, "epoch": epoch, "results": results})
        except:
            print("Count not process:", test_file)
            continue


    # Aggregated Results
    df_eval = pd.DataFrame(eval_dict_list)
    df_test = pd.DataFrame(test_dict_list)

    # Save Results
    df_eval.to_csv(os.path.join(data_dir, final_eval_file), index=False, sep='\t')
    df_test.to_csv(os.path.join(data_dir, final_test_file), index=False, sep='\t')


    ############################################################################
    #                         DEV ACCURACY VS. EPOCH                           #
    ############################################################################

    create_directory(plots_dir)

    plt.figure()

    # Plot Baselines
    xlim = [min(df_eval["epoch"]), max(df_eval["epoch"])]
    plt.plot(xlim, [0.829]*2, 'r--', label=r"$\mathrm{Top \, System}$")
    plt.plot(xlim, [0.800]*2, 'k--', label=r"$\mathrm{CNN}$")
    plt.plot(xlim, [0.750]*2, 'g--', label=r"$\mathrm{BiLSTM}$")
    plt.plot(xlim, [0.690]*2, 'b--', label=r"$\mathrm{SVM}$")

    # Plot Results
    for f in np.unique(df_eval['fold']):
        df_fold = df_eval[df_eval['fold']==f].sort_values(by=['epoch'])
        plt.plot(df_fold["epoch"], df_fold["eval_accuracy"],
                 label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1)

    # Formatting
    plt.xlim(xlim)
    plt.ylim(0.65, 0.85) # Empirically determined
    plt.legend(loc='lower center', prop={'size': 12}, ncol=3)
    plt.xlabel(r"$\mathrm{Epochs}$")
    plt.ylabel(r"$\mathrm{Dev \, Accuracy}$")

    # Save Figure
    plt.savefig(os.path.join(plots_dir, plot_eval), bbox_inches = 'tight')
    plt.close()


    ############################################################################
    #         Comparison between Different Pairwise Similarity Metrics         #
    ############################################################################

    threshold = 0.5

    # Track Plots
    fig1, ax1 = plt.subplots()  # Dev Accuracy
    fig2, ax1 = plt.subplots()  # Dev F1-Macro
    fig3, ax1 = plt.subplots()  # Dev F1-Micro
    fig4, ax1 = plt.subplots()  # Dev F1-Weighted
    fig5, ax1 = plt.subplots()  # Cos Sim (Pred vs. Prob)
    fig6, ax1 = plt.subplots()  # Cos Sim (Pred) vs. Overlap
    fig7, ax1 = plt.subplots()  # Cos Sim (Prob) vs. Overlap

    # Define Color Parameters
    folds = np.unique(df_test['fold'])
    NUM_COLORS = folds.size
    cm = plt.get_cmap('gist_rainbow')


    for f in folds:
        
        # Probabilities & Predictions
        df_results  = df_test[df_test['fold']==f].sort_values(by=['epoch'])[['epoch','results']]
        epochs      = df_results['epoch']
        probability = np.array([row for row in df_results['results']])
        predictions = 1*(probability>threshold)

        # Gold Labels
        labels = np.genfromtxt(os.path.join(data_dir, "fold{0}".format(f), label_file))
        if predictions.shape[1] == labels.shape:
            gold_matrix = np.tile(labels, [predictions.size // labels.size, 1])
        else:
            # Normally, this isnt needed, but I initially forgot to include the
            # 1 line header in the test.tsv files, so there's an off-by-1 error
            # in some folds.
            gold_matrix = np.tile(labels[1:], [predictions.size // (labels.size-1), 1])

        # Similarity Metrics  (Matrix of Size [#epochs x #epochs])
        cos_sim_pred = sm.cosine_similarity(predictions)
        cos_sim_prob = sm.cosine_similarity(probability)
        overlap_pred = sm.overlap_percentage(predictions)

        # Evaluation Metrics  (Vector of Size [#epochs])
        accuracy   = em.accuracy( predictions, gold_matrix )
        f1micro    = em.f1_metric(predictions, gold_matrix, average='micro')
        f1macro    = em.f1_metric(predictions, gold_matrix, average='macro')
        f1weighted = em.f1_metric(predictions, gold_matrix, average='weighted')

        ax1.plot(epochs, accuracy)
        ax2.plot(epochs, f1micro)
        ax3.plot(epochs, f1macro)
        ax4.plot(epochs, f1weighted)

        # Comparison between Similarity Metrics
        ax5.scatter(cos_sim_pred, cos_sim_prob, s=0.5)
        ax6.scatter(cos_sim_pred, 100*overlap_pred, s=0.5)
        ax7.scatter(cos_sim_prob, 100*overlap_pred, s=0.5)


    ax1.legend(loc='lower center', prop={'size': 12}, ncol=3)
    ax1.set_xlabel(r"$\mathrm{Epochs}$")
    ax1.set_ylabel(r"$\mathrm{Dev \, Accuracy}$")
    ax1.close()
    
    ax2.close()
    
    ax3.close()
    

    ax4.close()
    
    ax5.set_xlabel(r"$\mathrm{Cosine \, Similarity \, (Predictions)}$")
    ax5.set_ylabel(r"$\mathrm{Cosine \, Similarity \, (Probability)}$")
    ax5.savefig(os.path.join(plots_dir, plot_cospred_cosprob), bbox_inches = 'tight')
    ax5.close()
    
    ax6.set_xlabel(r"$\mathrm{Cosine \, Similarity \, (Predictions)}$")
    ax6.set_ylabel(r"$\mathrm{Overlap \, (\%)}$")
    ax6.savefig(os.path.join(plots_dir, plot_cospred_overlap), bbox_inches = 'tight')
    ax6.close()
    
    ax7.xlabel(r"$\mathrm{Cosine \, Similarity \, (Probability)}$")
    ax7.set_ylabel(r"$\mathrm{Overlap \, (\%)}$")
    ax7.savefig(os.path.join(plots_dir, plot_cosprob_overlap), bbox_inches = 'tight')
    ax7.close()

    # Notes on Training:
    # Currently training fold 5.
    # Need ot finish predicting fold 4.
    # Retar fold 4 results.
    # Need to change predict_bert.sh script to fold 5 after...


################################################################################

if __name__ == '__main__':
    main()


