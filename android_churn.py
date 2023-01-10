from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH, generate_random_temporal_features
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips
from utils.visualization import plot_results_sequence_svm

from utils.android_trainer import train_sequence_svm
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def main_train_sequence_svm():
    C_list = [0.01]
    max_iter = 1000
    class_weight = 'balanced'
    # sample_weight_list = [None, 2, 5, 10, 20, 50, 100]
    sample_weight_list = [5]
    overwrite = True
    test_size = 3
    train_size = 12
    n_updates = 10

    fname = f"results_cw-{class_weight}_tr-{train_size}_ts-{test_size}_C-{C_list}"
    results_path = f"results/android/{fname}.pkl"
    fig_fname = 'android_temporal_churn'

    # # if not os.path.exists(results_path) or overwrite:
    # train_sequence_svm(results_path=results_path,
    #                    train_size=train_size,
    #                    test_size=test_size,
    #                    n_updates=n_updates,
    #                    C_list=C_list, class_weight=class_weight,
    #                    sample_weight_list=sample_weight_list, max_iter=max_iter,
    #                    overwrite=overwrite)

    plot_results_sequence_svm(results_path=results_path,
                              fig_fname=fig_fname)




if __name__ == "__main__":
    main_train_sequence_svm()


    print("")






