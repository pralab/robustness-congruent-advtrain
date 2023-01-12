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
    C = 0.01
    max_iter = 1000
    class_weight = 'balanced'
    sample_weight_list = [None, 2, 5, 10, 100, 1000]
    # sample_weight_list = [5]
    overwrite = True
    test_size = 5
    train_size = 12
    n_updates = 10
    temporal_weight = True

    for C in [0.001, 0.01, 0.1, 1]:
        for temporal_weight in [False, True]:
            fname = f"results_temporal-{temporal_weight}_cw-{class_weight}_tr-{train_size}_ts-{test_size}_C-{C}"
            # fname = "results_cw-balanced_tr-12_ts-3_C-[0.01]"
            results_path = f"results/android/new/{fname}.pkl"
            fig_fname = fname

            # # if not os.path.exists(results_path) or overwrite:
            train_sequence_svm(results_path=results_path,
                               train_size=train_size,
                               test_size=test_size,
                               n_updates=n_updates,
                               C=C, class_weight=class_weight,
                               sample_weight_list=sample_weight_list,
                               temporal_weight=temporal_weight,
                               max_iter=max_iter,
                               overwrite=overwrite)

            plot_results_sequence_svm(results_path=results_path,
                                      fig_fname=fig_fname)




if __name__ == "__main__":
    main_train_sequence_svm()


    print("")






