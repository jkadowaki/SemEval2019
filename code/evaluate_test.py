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
import evaluate_results as er


################################################################################

def main(verbose=False):
    
    ############################################################################
    #                            FILENAME CONVENSION                           #
    ############################################################################
    
    # Directories
    data_dir        = 'test_data'
    results_dir     = 'epoch_results'
    
    # Constants
    steps_in_epoch  = 373

    # Generated Files
    final_test_file          = 'test_results.csv'
    eval_metrics_file        = 'eval_metrics.csv'
    eval_metrics_pickle_file = 'eval_metrics.pkl'
    
    ############################################################################
    #         AGGREGATE ALL INDIVIDUAL RESULTS INTO A SINGLE DATAFRAME         #
    ############################################################################

    # Crawl the Data Directory for Eval & Test Results Files
    test_files = er.get_results(data_dir, prefix='test_results_', extension='.tsv')
    
    # Temporarily Data Storage for All Results Files
    test_dict_list = []


    for test_file in test_files:
        # Get the Epoch Number from the File Path
        epoch = int( os.path.splitext(os.path.basename(
                       test_file))[0].split('_')[2] ) // steps_in_epoch
       
        # Get Prediction
        try:
            results = np.genfromtxt(test_file, unpack=True, encoding='bytes')[1]
            test_dict_list.append({"epoch": epoch, "results": results})
        except:
            print("Count not process:", test_file)
            continue


    # Aggregated Results
    df_test = pd.DataFrame(test_dict_list)

    # Save Results
    df_test.to_csv(os.path.join(data_dir, final_test_file), index=False, sep='\t')

    results_dict = []


    ########################################################################
    #                              Probabilities                           #
    ########################################################################
    
    # Probabilities & Predictions
    df_results  = df_test.sort_values(by=['epoch'])[['epoch','results']]
    epochs      = df_results['epoch']
    probability = np.array([row for row in df_results['results']])


    # After running the code up to this point, we opted for for this threshold value.
    threshold = 0.4
    predictions = 1*(probability>threshold)


    results_dict = []

    for idx,e in enumerate(epochs):
        # Save Computation
        results_dict.append({"epoch":       e,
                             "prediction":  list(predictions[idx]),
                             "probability": list(probability[idx])} )

    # Saving Evaluation Results
    df_eval_metrics = pd.DataFrame(results_dict)
    df_eval_metrics.to_csv(os.path.join(data_dir, eval_metrics_file),
                           index=False, sep='\t')
    df_eval_metrics.to_pickle(os.path.join(data_dir, eval_metrics_pickle_file))

    ########################################################################
    #                              Prediction                              #
    ########################################################################

    from sklearn.metrics import accuracy_score, f1_score
    ens_epochs=[2,3,4]
    prediction = np.vstack(df_eval_metrics[
                    df_eval_metrics['epoch'].isin(ens_epochs)].prediction.values)
    probability = np.vstack(df_eval_metrics[
                    df_eval_metrics['epoch'].isin(ens_epochs)].probability.values)

    ensemble_prediction = (np.sum(prediction, axis=0)/np.size(ens_epochs) > 0.5).astype(int)
    ensemble_probability = (np.sum(probability, axis=0)/np.size(ens_epochs) > 0.4).astype(int)
    gl = np.array(df_eval_metrics.gold_label.values[0])
    f1_score(ensemble_prediction, gl, average='macro')
    f1_score(ensemble_probability, gl, average='macro')

    ensemble_labels = ['OFF' if i==1 else 'NOT' for i in ensemble_prediction]
    df_final = pd.read_csv('OLIDv1.0/testset-levela.tsv', sep='\t')
    df_final['label']=ensemble_labels
    df_final[['id','label']].to_csv('test_data/labels-levela.csv',header=False, index=False)

################################################################################

if __name__ == '__main__':
    main()


