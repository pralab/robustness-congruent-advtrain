import os
from utils.visualization import plot_sequence_results_android
from itertools import product as cartesian_product
from utils.android_trainer import AndroidTemporalTrainer


def main_train_sequence():

    clf_info = {'clf_name': 'svm',
                'C': 0.01,
                'max_iter': 1000}

    # clf_info = {'clf_name': 'rf',
    #             'n_estimators': 20,
    #             'max_depth': 10}

    class_weight = 'balanced'
    # sample_weight_list = [None, 2, 5, 10, 100, 1000]
    sample_weight = [None, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100]
    overwrite = True
    test_size = 5
    train_size = 12
    val_size = 0 #1
    max_fpr = 0.01
    n_updates = 12
    temporal_weight = False


    class_weight_list = [None, 'balanced']
    max_fpr_list = [.01, .02, .05]

    prod = list(cartesian_product(class_weight_list, max_fpr_list))
    n_runs = len(prod)

    for i, (class_weight, max_fpr) in enumerate(prod):
        print(f"{i+1}/{n_runs}: cw: {class_weight}, max_fpr: {max_fpr}")

        fname = f"cw-{class_weight}_" \
                f"tr-{train_size}_" \
                f"ts-{test_size}_" \
                f"val-{val_size}_" \
                f"fpr-{max_fpr}_" \
                f"n_updates-{n_updates}_" \
                f"{clf_info['clf_name']}"
        # fname = "results_cw-balanced_tr-12_ts-3_C-[0.01]"
        results_path = f"results/android/{fname}.pkl"
        fig_fname = fname

        title = f"{clf_info}" \
                f"CW: {class_weight}, " \
                f"TR: {train_size}, " \
                f"TS: {test_size}, " \
                f"VAL: {val_size}, " \
                f"FPR: {max_fpr}, " \
                f"Updates: {n_updates}"

        if not os.path.exists(results_path) or overwrite:
            trainer = AndroidTemporalTrainer(results_path=results_path,
                                             train_size=train_size,
                                             test_size=test_size,
                                             val_size=val_size,
                                             max_fpr=max_fpr,
                                             n_updates=n_updates,
                                             clf_info=clf_info,
                                             class_weight=class_weight,
                                             sample_weight=sample_weight,
                                             temporal_weight=temporal_weight,
                                             overwrite=overwrite)
            trainer.train_sequence_parametric()


        plot_sequence_results_android(results_path=results_path,
                                  fig_fname=fig_fname,
                                  title=title)


if __name__ == "__main__":
    main_train_sequence()







