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

if __name__ == "__main__":

    # with open(os.path.join(DS_PATH, 'drebin_xyt.pkl'), 'rb') as f:
    #     ds = pickle.load(f)
    # X, y, t, m = ds['X'], ds['y'], ds['t'], ds['m']
    # y = np.array([int(y[0]) for y in y])
    # # Partition dataset
    # splits = temporal.time_aware_train_test_split(
    #    X, y, t, train_size=12, test_size=1, granularity='month')
    # X_train, X_tests, y_train, y_tests, t_train, t_tests, train, tests = splits
    # # y_train = [int(y[0]) for y in y_train]

    X, y = generate_random_temporal_features(n_samples_per_month=100,
                                             n_features=100,
                                             n_months=6)
    X_train_i = X[0]
    y_train_i = y[0]
    X_tests, y_tests = X[1:], y[1:]

    # X_train_i = X_train
    # y_train_i = y_train

    precs, recs, f1s = [], [], []
    nfrs_pos, pfrs_pos = [], []
    nfrs_neg, pfrs_neg = [], []

    sample_weights = None
    n_updates = len(X_tests) - 3
    for i in range(n_updates):
        print(f"> M{i} ({i}/{n_updates})")

        clf = LinearSVC()
        clf.fit(X_train_i, y_train_i, sample_weight=sample_weights)

        preds_tr = clf.predict(X_train_i)
        sample_weights = ((preds_tr == y_train_i)*2).astype(np.float_)

        X_test_i = vstack((X_tests[i], X_tests[i+1]))
        y_test_i = np.hstack((y_tests[i], y_tests[i+1]))
        preds = clf.predict(X_test_i)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_i, preds,
                                                           pos_label=1,
                                                           average='binary')
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        preds1 = preds[:X_tests[i].shape[0]]

        if i > 0:
            nfr = compute_nflips(old_preds=preds2, new_preds=preds1, indexes=True)
            pfr = compute_pflips(old_preds=preds2, new_preds=preds1, indexes=True)
            nfr_pos = nfr[y_tests[i] == 1].mean()*100
            nfr_neg = nfr[y_tests[i] == 0].mean()*100
            pfr_pos = pfr[y_tests[i] == 1].mean()*100
            pfr_neg = pfr[y_tests[i] == 0].mean()*100
        else:
            nfr_pos, nfr_neg, pfr_pos, pfr_neg = None, None, None, None

        nfrs_pos.append(nfr_pos)
        nfrs_neg.append(nfr_neg)
        pfrs_pos.append(pfr_pos)
        pfrs_neg.append(pfr_neg)

        preds2 = preds[X_tests[i].shape[0]:]
        X_train_i = vstack((X_train_i, X_tests[i]))
        y_train_i = np.hstack((y_train_i, y_tests[i]))


    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(f1s, color='blue', marker='o', label='F1')
    ax[0].plot(precs, color='green', marker='*', label='Precision')
    ax[0].plot(recs, color='red', marker='s', label='Recall')

    ax[1].plot(nfrs_pos, color='red', marker='v', label='NFR-mw')
    ax[1].plot(nfrs_neg, color='green', marker='^', label='NFR-gw')

    ax[2].plot(pfrs_pos, color='red', marker='>', label='PFR-mw')
    ax[2].plot(pfrs_neg, color='green', marker='<', label='PFR-gw')

    titles = ['Accuracy (%)',
              'Negative Flip Rate (%)',
              'Positive Flip Rate (%)']
    for i, title in enumerate(titles):
        ax[i].set_title(title)
        ax[i].set_xlabel('Updates')
        ax[i].set_xticks(np.arange(start=0, stop=n_updates, step=3))
        ax[i].legend()
    fig.tight_layout()
    fig.show()
    #fig.savefig(os.path.join('images/android', 'android_churn.pdf'))
    print("")






