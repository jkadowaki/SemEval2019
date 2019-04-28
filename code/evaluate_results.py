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

def main(verbose=False):
    
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
    final_eval_file   = 'eval_results.csv'
    final_test_file   = 'test_results.csv'
    eval_metrics_file = 'eval_metrics.csv'
    plot_eval            = "dev_results.pdf"
    plot_accuracy        = "accuracy.pdf"
    plot_f1micro         = "f1_micro.pdf"
    plot_f1macro         = "f1_macro.pdf"
    plot_f1weighted      = "f1_weighted.pdf"
    plot_cospred_cosprob = "cospred_cosprob.pdf"
    plot_cospred_overlap = "cospred_overlap.pdf"
    plot_cosprob_overlap = "cosprob_overlap.pdf"
    plot_threshold       = "threshold{0}.pdf"
    
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
    xlim = [min(df_eval["epoch"]), max(df_eval["epoch"])]

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

    # Track Plots
    fig1, ax1 = plt.subplots()  # Dev Accuracy vs. Epoch
    fig2, ax2 = plt.subplots()  # Dev F1-Macro vs. Epoch
    fig3, ax3 = plt.subplots()  # Dev F1-Micro vs. Epoch
    fig4, ax4 = plt.subplots()  # Dev F1-Weighted vs. Epoch
    fig5, ax5 = plt.subplots()  # Cos Sim (Pred vs. Cos Sim (Prob)
    fig6, ax6 = plt.subplots()  # Cos Sim (Pred) vs. Overlap
    fig7, ax7 = plt.subplots()  # Cos Sim (Prob) vs. Overlap

    # Define Color Parameters
    folds = np.unique(df_test['fold'])
    NUM_COLORS = folds.size
    cm = plt.get_cmap('gist_rainbow')

    # Plot Baselines
    ax2.plot(xlim, [0.829]*2, 'r--', label=r"$\mathrm{Top \, System}$")
    ax2.plot(xlim, [0.800]*2, 'g--', label=r"$\mathrm{CNN}$")
    ax2.plot(xlim, [0.750]*2, 'k--', label=r"$\mathrm{BiLSTM}$")
    ax2.plot(xlim, [0.690]*2, 'b--', label=r"$\mathrm{SVM}$")

    # Save Results
    results_dict = []


    for f in folds:
        
        ########################################################################
        #                      Probabilities & Gold Labels                     #
        ########################################################################
        
        # Probabilities & Predictions
        df_results  = df_test[df_test['fold']==f].sort_values(by=['epoch'])[['epoch','results']]
        epochs      = df_results['epoch']
        probability = np.array([row for row in df_results['results']])
        
        # Gold Labels
        labels = np.genfromtxt(os.path.join(data_dir, "fold{0}".format(f), label_file))
        if probability.shape[1] == labels.shape[0]:
            gold_matrix = np.tile(labels, [probability.size // labels.size, 1])
        else:
            # Normally, this isnt needed, but I initially forgot to include the
            # 1 line header in the test.tsv files, so there's an off-by-1 error
            # in some folds.
            gold_matrix = np.tile(labels[1:], [probability.size // (labels.size-1), 1])


        ########################################################################
        #                       Determining the Threshold                      #
        ########################################################################
        
        if verbose: print("\nFold", f)
        fig8, ax8 = plt.subplots()

        # Determine Threshold
        for t in np.arange(0.35,0.5,0.01):
            
            predictions = 1*(probability>t)
            f1macro = em.f1_metric(predictions, gold_matrix, average='macro')
            if verbose:
                print( "\tThreshold:{0}".format(round(t,2)),
                      #"\tF1:", f1macro[1:9],
                       "\tF1 (mean)", np.mean(f1macro[1:9]),
                       "\tF1 (max):", max(f1macro[1:9]) )
            
            lw=2 if round(t,2) in [0.45, 0.50] else 0.5
            ax8.plot(epochs[:10], f1macro[:10], linewidth=lw, label=str(round(t,2)))

        ax8.legend(loc='lower right', prop={'size': 12}, ncol=4)
        ax8.set_xlim(0,9)
        ax8.set_ylim(0.6, 0.8)
        ax8.set_xlabel(r"$\mathrm{Epochs}$")
        ax8.set_ylabel(r"$\mathrm{Macro-F1 \, Score}$")
        fig8.savefig(os.path.join(plots_dir, plot_threshold.format(f)), bbox_inches='tight')
        fig8.clf()
        
        # After running the code up to this point, we opted for for this threshold value.
        threshold = 0.45
        predictions = 1*(probability>threshold)


        ########################################################################
        #                              Evaluation                              #
        ########################################################################

        # Similarity Metrics  (Matrix of Size [#epochs x #epochs])
        cos_sim_pred = sm.cosine_similarity(predictions)
        cos_sim_prob = sm.cosine_similarity(probability)
        overlap_pred = sm.overlap_percentage(predictions)

        # Evaluation Metrics  (Vector of Size [#epochs])
        accuracy   = em.accuracy( predictions, gold_matrix )
        f1macro    = em.f1_metric(predictions, gold_matrix, average='macro')
        f1micro    = em.f1_metric(predictions, gold_matrix, average='micro')
        f1weighted = em.f1_metric(predictions, gold_matrix, average='weighted')

        for idx,e in enumerate(epochs):
            # Save Computation
            results_dict.append({"fold":        f,
                                 "epoch":       e,
                                 "prediction":  list(predictions[idx]),
                                 "probability": list(probability[idx]),
                                 "accuracy":    list(accuracy),
                                 "f1_macro":    list(f1macro),
                                 "f1_micro":    list(f1micro),
                                 "f1_weighted": list(f1weighted),
                                 "gold_label": list(gold_matrix[idx])} )

        # Plot: Evaluation Metrics vs. Epochs
        ax1.plot( epochs, accuracy,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax2.plot( epochs, f1macro,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax3.plot( epochs, f1micro,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )
        ax4.plot( epochs, f1weighted,
                  label=r"$\mathrm{{Fold \, {0} }}$".format(f), linewidth=1 )

        # Plot: Comparison between Similarity Metrics
        ax5.scatter( cos_sim_pred, cos_sim_prob,     marker='.', s=0.1,
                     label=r"$\mathrm{{Fold \, {0} }}$".format(f) )
        ax6.scatter( cos_sim_pred, 100*overlap_pred, marker='.', s=0.1,
                     label=r"$\mathrm{{Fold \, {0} }}$".format(f) )
        ax7.scatter( cos_sim_prob, 100*overlap_pred, marker='.', s=0.1,
                     label=r"$\mathrm{{Fold \, {0} }}$".format(f) )


    # Saving Evaluation Results
    df_eval_metrics = pd.DataFrame(results_dict)
    df_eval_metrics.to_csv(os.path.join(data_dir, eval_metrics_file),
                           index=False, sep='\t')

    # Dev Accuracy vs. Epoch
    ax1.legend(loc='lower center', prop={'size': 12}, ncol=3)
    ax1.set_xlim(0,100)
    ax1.set_ylim(0.725, 0.85)
    ax1.set_xlabel(r"$\mathrm{Epochs}$")
    ax1.set_ylabel(r"$\mathrm{Accuracy}$")
    fig1.savefig(os.path.join(plots_dir, plot_accuracy), bbox_inches = 'tight')
    fig1.clf()
    
    # Dev F1-Macro vs. Epoch
    ax2.legend(loc='lower center', prop={'size': 12}, ncol=3)
    ax2.set_xlim(0,100)
    ax2.set_ylim(0.65, 0.83)
    ax2.set_xlabel(r"$\mathrm{Epochs}$")
    ax2.set_ylabel(r"$\mathrm{Macro-F1 \, Score}$")
    fig2.savefig(os.path.join(plots_dir, plot_f1macro), bbox_inches='tight')
    fig2.clf()
    
    # Dev F1-Micro vs. Epoch
    ax3.legend(loc='lower center', prop={'size': 12}, ncol=3)
    ax3.set_xlim(0,100)
    ax3.set_ylim(0.725, 0.85)
    ax3.set_xlabel(r"$\mathrm{Epochs}$")
    ax3.set_ylabel(r"$\mathrm{Micro-F1 \, Score}$")
    fig3.savefig(os.path.join(plots_dir, plot_f1micro), bbox_inches='tight')
    fig3.clf()
    
    # Dev F1-Weighted vs. Epoch
    ax4.legend(loc='lower center', prop={'size': 12}, ncol=3)
    ax4.set_xlim(0,100)
    ax4.set_ylim(0.725, 0.85)
    ax4.set_xlabel(r"$\mathrm{Epochs}$")
    ax4.set_ylabel(r"$\mathrm{Weighted-F1 \, Score}$")
    fig4.savefig(os.path.join(plots_dir, plot_f1weighted), bbox_inches='tight')
    fig4.clf()
    
    # Cos Sim (Pred vs. Cos Sim (Prob)
    ax5.plot([0,0], [1,1], 'k-', linewidth=1)
    ax5.set_xlim(0,1)
    ax5.set_ylim(0.45,1)
    ax5.set_xlabel(r"$\mathrm{Cosine \, Similarity \, (Predictions)}$")
    ax5.set_ylabel(r"$\mathrm{Cosine \, Similarity \, (Probability)}$")
    fig5.savefig(os.path.join(plots_dir, plot_cospred_cosprob), bbox_inches='tight')
    fig5.clf()
    
    # Cos Sim (Pred) vs. Overlap
    ax6.set_xlim(0,1)
    ax6.set_ylim(0,100)
    ax6.set_xlabel(r"$\mathrm{Cosine \, Similarity \, (Predictions)}$")
    ax6.set_ylabel(r"$\mathrm{Overlap \, (\%)}$")
    fig6.savefig(os.path.join(plots_dir, plot_cospred_overlap), bbox_inches='tight')
    fig6.clf()
    
    # Cos Sim (Prob) vs. Overlap
    ax7.set_xlim(0.45,1)
    ax7.set_ylim(0,100)
    ax7.set_xlabel(r"$\mathrm{Cosine \, Similarity \, (Probability)}$")
    ax7.set_ylabel(r"$\mathrm{Overlap \, (\%)}$")
    fig7.savefig(os.path.join(plots_dir, plot_cosprob_overlap), bbox_inches='tight')
    fig7.clf()


################################################################################

if __name__ == '__main__':
    main()


