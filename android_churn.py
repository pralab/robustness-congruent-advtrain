from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH, generate_random_temporal_features
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def ds_stack(X: list, y:list,
             start: int = 0, n_months: int = 12):
    X_stack = vstack(X[start: start + n_months])
    y_stack = np.hstack(y[start: start + n_months])

    idx = 0
    idxs = []
    for x in X[start: start + n_months]:
        idx += x.shape[0]
        idxs.append(idx)
    #
    # X_unstack, y_unstack = ds_unstack(X_stack, y_stack, idxs)
    #
    # s = 0
    # for i in range(n_months):
    #     s += (X_unstack[i] != X[start: start + n_months][i]).nnz
    #     s += (y_unstack[i] != y[start: start + n_months][i]).sum()
    # assert s == 0

    return X_stack, y_stack, idxs

def ds_unstack(X: csr_matrix, y: np.ndarray, idxs: list, preds: list):
    X_unstack = [X[0 if i == 0 else idxs[i-1]: idxs[i]] for i in range(len(idxs))]
    y_unstack = np.split(y, idxs)[:-1]
    preds_unstack = np.split(preds, idxs)[:-1]
    return X_unstack, y_unstack, preds_unstack


def train_sequence_svm(results_path, train_size, test_size,
                       C_list=(0.001, 0.01, 0.1),
                       class_weight='balanced',
                       sample_weights=None,
                       max_iter=1000,
                       overwrite=False
                       ):

    if not os.path.exists(results_path) or overwrite:
        with open(os.path.join(DS_PATH, 'drebin_xyt.pkl'), 'rb') as f:
            ds = pickle.load(f)
        X, y, t, m = ds['X'], ds['y'], ds['t'], ds['m']
        y = np.array([int(y[0]) for y in y])
        # Partition dataset
        # qui prendo il dataset splittato in mesi, senza fare ancora train-test split
        _, X, _, y, *_ = temporal.time_aware_train_test_split(
           X, y, t, train_size=0, test_size=1, granularity='month')

        # X, y = generate_random_temporal_features(n_samples_per_month=100,
        #                                          n_features=100,
        #                                          n_months=10)

        results = []
        for row, C in enumerate(C_list):
            t = f" C={C} "
            print(f"{t:#^40}")

            # X_train_i = X[0]
            # y_train_i = y[0]
            # X_tests, y_tests = X[1:], y[1:]

            precs, recs, f1s = [], [], []
            nfrs_pos, pfrs_pos = [], []
            nfrs_neg, pfrs_neg = [], []

            n_updates = len(X) - train_size - test_size
            for i in range(n_updates):
                print(f"\n> M{i}/{n_updates}")

                # Obtain train window
                X_train_i, y_train_i, train_idxs = ds_stack(X, y, start=i, n_months=train_size)
                print(f"Train months: {len(train_idxs)}, N samples: {X_train_i.shape[0]}")
                clf = LinearSVC(C=C,
                                class_weight=class_weight,
                                max_iter=max_iter)
                clf.fit(X_train_i,
                        y_train_i,
                        sample_weight=sample_weights
                        )
                # clf.fit(X_train_i[:10],
                #         y_train_i[:10],
                #         sample_weight=sample_weights,
                #         )

                # # Churn-aware filter
                # preds_tr = clf.predict(X_train_i)
                # sample_weights = np.ones(preds_tr.shape)
                # sample_weights[preds_tr == y_train_i] = 2

                # Obtain test window
                X_test_i, y_test_i, test_idxs = ds_stack(X, y,
                                                         start=train_size + i,
                                                         n_months=test_size)
                print(f"Test months: {len(test_idxs)}, N samples: {X_test_i.shape[0]}")
                preds = clf.predict(X_test_i)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test_i, preds,
                                                                   pos_label=1,
                                                                   average='binary')
                precs.append(prec*100)
                recs.append(rec*100)
                f1s.append(f1*100)

                # preds1 = preds[:X_tests[i].shape[0]]

                _, y_test_splitted, preds_splitted = ds_unstack(X_test_i, y_test_i, test_idxs, preds)

                if i > 0:
                    new_preds = np.hstack(preds_splitted[:-1])
                    y_common = np.hstack(y_test_splitted[:-1])
                    nfr = compute_nflips(old_preds=old_preds, new_preds=new_preds, indexes=True)
                    pfr = compute_pflips(old_preds=old_preds, new_preds=new_preds, indexes=True)
                    nfr_pos = nfr[y_common == 1].mean()*100
                    nfr_neg = nfr[y_common == 0].mean()*100
                    pfr_pos = pfr[y_common == 1].mean()*100
                    pfr_neg = pfr[y_common == 0].mean()*100
                else:
                    nfr_pos, nfr_neg, pfr_pos, pfr_neg = None, None, None, None

                nfrs_pos.append(nfr_pos)
                nfrs_neg.append(nfr_neg)
                pfrs_pos.append(pfr_pos)
                pfrs_neg.append(pfr_neg)

                old_preds = np.hstack(preds_splitted[1:])

            results.append(
                {
                    'f1s': f1s,
                    'precs': precs,
                    'recs': recs,
                    'nfrs_pos': nfrs_pos,
                    'nfrs_neg': nfrs_neg,
                    'pfrs_pos': pfrs_pos,
                    'pfrs_neg': pfrs_neg,
                    'C': C,
                    'class_weight': class_weight,
                    'sample_weights': sample_weights
                }
            )

        with open(os.path.join(results_path), 'wb') as f:
            pickle.dump(results, f)


def main_train_sequence_svm():
    C_list = [0.001, 0.01, 0.1, 1]
    max_iter = 1000
    class_weight = 'balanced'
    sample_weights = None
    overwrite = True
    test_size = 3
    train_size = 12

    fname = f"results_cw-{class_weight}_tr-{train_size}_ts-{test_size}_C-{C_list}"
    results_path = f"results/android/{fname}.pkl"

    ##############################
    # Train
    ##############################
    train_sequence_svm(results_path=results_path,
                       train_size=train_size,
                       test_size=test_size,
                       C_list=C_list, class_weight=class_weight,
                       sample_weights=sample_weights, max_iter=max_iter,
                       overwrite=True)


def plot_result(result, ax, row=0):
    ax[row, 0].plot(result['f1s'], color='blue', marker='o', label='F1')
    ax[row, 0].plot(result['precs'], color='green', marker='*', label='Precision')
    ax[row, 0].plot(result['recs'], color='red', marker='s', label='Recall')

    ax[row, 1].plot(result['nfrs_pos'], color='red', marker='v', label='NFR-mw')
    ax[row, 1].plot(result['nfrs_neg'], color='green', marker='^', label='NFR-gw')

    # ax[row, 2].plot(result['pfrs_pos'], color='red', marker='>', label='PFR-mw')
    # ax[row, 2].plot(result['pfrs_neg'], color='green', marker='<', label='PFR-gw')
    # ax[row, 0].set_ylabel(f"C = {result['C']}")

    # titles = ['Performances (%)',
    #           'Negative Flip Rate (%)',
    #           'Positive Flip Rate (%)']
    titles = ['Performances (%)',
              'Negative Flip Rate (%)']
    for i, title in enumerate(titles):
        ax[row, i].set_title(title)
        ax[row, i].set_xlabel('Updates')
        ax[row, i].set_xticks(np.arange(start=0, stop=len(result['f1s']), step=3))
        ax[row, i].legend()


def plot_results_sequence_svm(results_path, results_fname,
                              fig_fname):

    with open(os.path.join(results_path, results_fname), 'rb') as f:
        results = pickle.load(f)

    for res in results:
        if res['C'] == 0.01:
            result = res
            break

    # fig, ax = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))
    # for row, result in enumerate(results):
    #     plot_result(result, ax, row)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    plot_result(result, ax)

    fig.tight_layout()
    fig.show()
    fig.savefig(f"images/android/{fig_fname}.pdf")


if __name__ == "__main__":
    results_path = 'results/android'
    results_fname = 'results_cw-balanced_tr-12_ts-5_C-[0.001, 0.01, 0.1, 1].pkl'
    fig_fname = 'android_temporal_churn'
    ##############################
    # Plots
    ##############################
    plot_results_sequence_svm(results_path=results_path,
                              results_fname=results_fname,
                              fig_fname=fig_fname)
    print("")






