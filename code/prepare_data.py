#!/usr/bin/env python

from __future__ import print_function
import csv
import errno
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


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

def main():

    orig_data_dir   = 'OLIDv1.0'
    training_file   = 'olid-training-v1.0.tsv'
    test_text_file  = 'testset-levela.tsv'
    test_label_file = 'labels-levela.csv'

    data_directory    = 'eval_data'
    test_directory    = 'test_data'
    seed              = 100
    folds             = 10
    bert_train_file   = 'train.tsv'
    bert_dev_file     = 'dev.tsv'
    bert_test_file    = 'test.tsv'
    bert_label_file   = 'label.tsv'
    formatted_label   = 'output.csv'


    # Read in OLID Data Files
    olid_data = pd.read_csv( os.path.join(orig_data_dir, training_file),  sep='\t')
    olid_test = pd.read_csv( os.path.join(orig_data_dir, test_text_file), sep='\t')
    olid_gold = pd.read_csv( os.path.join(orig_data_dir, test_label_file),
                             sep=',', header=None )


    # Convert Text Labels to Integers
    encoder = LabelEncoder()
    binary_label = encoder.fit_transform(olid_data['subtask_a'])
    test_label   = encoder.transform(olid_gold[1])

    # Annotated Training & Dev Sets
    d = { 'id':    olid_data['id'],        # Column 1: Tweet ID
          'label': binary_label,           # Column 2: Label (INT)
          'alpha': ['a']*len(olid_data),   # Column 3: A column of all the same letter
          'tweet': olid_data['tweet'] }    # Column 4: Text to Classify in Training
    df_bert = pd.DataFrame(data=d)

    # Create a Directory to Store Training/Dev/Test Sets
    create_directory(data_directory)
    create_directory(test_directory)

    # Create an Object to Perform Stratified K-fold CV
    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)


    ############################################################################
    #        Creates a Training/Dev (i.e., test=dev) Set for All Folds         #
    ############################################################################
    
    for idx, [train_index, dev_index] in enumerate(
            skf.split(df_bert[['id', 'alpha', 'tweet']] , df_bert[['label']]) ):
        
        # Stores Training & Dev Sets for Each Fold in Respective Directory
        fold_dir = 'fold{0}'.format(idx)
        create_directory(os.path.join(data_directory, fold_dir))
        
        # Splits the Annotated Dataset into Training & Dev Sets for Each Fold
        print("TRAIN:", train_index, "DEV:", dev_index)
        df_bert_train = df_bert.iloc[list(train_index)]
        df_bert_dev   = df_bert.iloc[list(dev_index)]

        # Save Training Set: "eval_data/fold<#>/train.tsv"
        df_bert_train.to_csv(os.path.join(data_directory, fold_dir, bert_train_file),
                             sep='\t', index=False, header=False)
        
        # Save Dev Set: "eval_data/fold<#>/dev.tsv"
        df_bert_dev.to_csv(os.path.join(data_directory, fold_dir, bert_dev_file),
                           sep='\t', index=False, header=False)
        
        # Save Test Set: "eval_data/fold<#>/test.tsv"
        df_bert_dev[['id','tweet']].to_csv(
                        os.path.join(data_directory, fold_dir, bert_test_file),
                        sep='\t', index=False, header=['id', 'tweet'] )

        # Save Test Set: "eval_data/fold<#>/test.tsv"
        df_bert_dev[['label']].to_csv(
                        os.path.join(data_directory, fold_dir, bert_label_file),
                        sep='\t', index=False, header=False )


    ############################################################################
    #      Prepares the Training/Dev/Test with ALL Data for Final Model        #
    ############################################################################

    # Save Training Set: "test_data/train.tsv"
    df_bert.to_csv( os.path.join(test_directory, bert_train_file),
                    sep='\t', index=False, header=False )

    # Save Test Set: "test_data/test.tsv"
    olid_test.to_csv( os.path.join(test_directory, bert_test_file),
                      sep='\t', index=False, header=['id', 'tweet'] )

    # Save Dev Set: "test_data/dev.tsv"
    d = { 'id':    olid_test['id'],        # Column 1: Tweet ID
          'label': test_label,             # Column 2: Label (INT)
          'alpha': ['a']*len(test_label),  # Column 3: A column of all the same letter
          'tweet': olid_test['tweet'] }    # Column 4: Text to Classify in Training
    df_olid = pd.DataFrame(data=d)

    df_olid.to_csv( os.path.join(test_directory, bert_dev_file),
                    sep='\t', index=False, header=False )


    # Save Labels (for Evaluation): "test_data/label.tsv"
    df_olid['label'].to_csv( os.path.join(test_directory, bert_label_file),
                             sep='\t', index=False, header=False )

    # Save Formatted Labels (as Sample Output): "test_data/output.csv"
    olid_gold.to_csv( os.path.join(test_directory, formatted_label),
                      sep=',', index=False, header=False )


################################################################################

if __name__ == '__main__':
    main()
