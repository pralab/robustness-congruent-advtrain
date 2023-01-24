import os
from utils.visualization import plot_sequence_results_android
from itertools import product as cartesian_product
from utils.android_trainer import AndroidTemporalTrainer

from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

def grid_search(root, global_param_dict):
    ordered_params = OrderedDict(global_param_dict)
    all_cross_validation_parameters = cartesian_product(*ordered_params.values())
    names = list(ordered_params.keys())

    partial_f = partial(train_single, root=root)
    with Pool(processes=1) as p:
        p.map(partial_f, all_cross_validation_parameters)


def train_single(clf_name, root, test_size=5, train_size=12, val_size=0, max_fpr=0.01,
                 n_updates=12, sample_weight=None, class_weight=None,
                 temporal_weight=False, overwrite=False,
                 n_estimators=10, max_depth=5,
                 C=0.01, max_iter=1000):


    fname = f"cw-{class_weight}_" \
            f"tr-{train_size}_" \
            f"ts-{test_size}_" \
            f"val-{val_size}_" \
            f"fpr-{max_fpr}_" \
            f"n_updates-{n_updates}_" \
            f"{clf_name}"
    if clf_name == 'rf':
        fname += f"_n_estim-{n_estimators}_" \
                 f"max_depth-{max_depth}"
        clf_info = {'clf_name': clf_name,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth}
    elif clf_name == 'svm':
        fname += f"_C-{C}_" \
                 f"max_iter-{max_iter}"
        clf_info = {'clf_name': clf_name,
                    'C': C,
                    'max_iter': max_iter}

    print(clf_info)
    # results_path = os.path.join(root, f"{fname}.pkl")
    # if not os.path.isdir(root):
    #     os.makedirs(root)
    #
    # fig_fname = fname
    #
    # title = f"{clf_info}" \
    #         f"CW: {class_weight}, " \
    #         f"TR: {train_size}, " \
    #         f"TS: {test_size}, " \
    #         f"VAL: {val_size}, " \
    #         f"FPR: {max_fpr}, " \
    #         f"Updates: {n_updates}"
    #
    # if not os.path.exists(results_path) or overwrite:
    #     trainer = AndroidTemporalTrainer(results_path=results_path,
    #                                      train_size=train_size,
    #                                      test_size=test_size,
    #                                      val_size=val_size,
    #                                      max_fpr=max_fpr,
    #                                      n_updates=n_updates,
    #                                      clf_info=clf_info,
    #                                      class_weight=class_weight,
    #                                      sample_weight=sample_weight,
    #                                      temporal_weight=temporal_weight,
    #                                      overwrite=overwrite)
    #     trainer.train_sequence_parametric()

    # plot_sequence_results_android(results_path=results_path,
    #                               fig_fname=fig_fname,
    #                               title=title)

def main_train_sequence():

    root = "results/android/prova"

    global_param_dict = {
        'clf_name': ['svm'],
        'C': [0.01], #[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        'max_iter': [1000],
        # 'clf_name': 'rf',
        # 'n_estimators': [10, 50, 100, 1000],
        # 'max_depth': [10, 20, 50, 100],
        'overwrite': [False],
        'test_size': [5],
        'train_size': [12],
        'val_size': [0], #[0, 1, 2],
        'n_updates': [12],
        # temporal_weight = False
        'class_weight': ['balanced', None],
        'sample_weight': [[None, 2, 5, 10, 100]],
        'max_fpr': [0.01],  #[0.01, 0.02, 0.05]
    }

    grid_search(root, global_param_dict)


if __name__ == "__main__":
    main_train_sequence()







