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



################################################################################

def main(verbose=False):
    
    # Directories
    data_dir  = 'eval_data'
    plots_dir = 'plots'

    # Data File
    eval_metrics_file = 'eval_metrics.csv'
    
    # Generated Files
    plot_accuracy        = "accuracy.pdf"
    plot_f1micro         = "f1_micro.pdf"
    plot_f1macro         = "f1_macro.pdf"
    plot_f1weighted      = "f1_weighted.pdf"
    plot_cospred_cosprob = "cospred_cosprob.pdf"
    plot_cospred_overlap = "cospred_overlap.pdf"
    plot_cosprob_overlap = "cosprob_overlap.pdf"

    
    ############################################################################
    #         AAA         #
    ############################################################################






################################################################################

if __name__ == '__main__':
    main()


