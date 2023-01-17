from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH, generate_random_temporal_features
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips
from utils.visualization import plot_results_sequence_svm

from utils.android_trainer import AndroidTemporalTrainer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def main_train_sequence_svm():
    C = 0.01
    max_iter = 1000
    class_weight = None #'balanced'
    # sample_weight_list = [None, 2, 5, 10, 100, 1000]
    sample_weight = [None, 2, 10, 100]
    overwrite = False
    test_size = 5
    train_size = 12
    val_size = 1
    fpr = 0.01
    n_updates = 12
    temporal_weight = False

    fname = f"results_temporal-{temporal_weight}_" \
            f"cw-{class_weight}_" \
            f"sw-{sample_weight}_" \
            f"tr-{train_size}_" \
            f"ts-{test_size}_" \
            f"val-{val_size}_" \
            f"fpr-{fpr}_" \
            f"n_updates-{n_updates}_" \
            f"C-{C}"
    # fname = "results_cw-balanced_tr-12_ts-3_C-[0.01]"
    results_path = f"results/android/{fname}.pkl"
    fig_fname = fname

    if not os.path.exists(results_path) or overwrite:
        trainer = AndroidTemporalTrainer(results_path=results_path,
                                         train_size=train_size,
                                         test_size=test_size,
                                         val_size=val_size,
                                         fpr=fpr,
                                         n_updates=n_updates,
                                         C=C, class_weight=class_weight,
                                         sample_weight=sample_weight,
                                         temporal_weight=temporal_weight,
                                         max_iter=max_iter,
                                         overwrite=overwrite)
        trainer.train_sequence_parametric()


    plot_results_sequence_svm(results_path=results_path,
                              fig_fname=fig_fname)




if __name__ == "__main__":
    main_train_sequence_svm()


    print("")






