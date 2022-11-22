from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips
from scipy.sparse import vstack
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open(os.path.join(DS_PATH, 'drebin_xyt.pkl'), 'rb') as f:
        ds = pickle.load(f)
    X, y, t, m = ds['X'], ds['y'], ds['t'], ds['m']
    y = np.array([int(y[0]) for y in y])
    # Partition dataset
    splits = temporal.time_aware_train_test_split(
       X, y, t, train_size=12, test_size=2, granularity='month')
    X_train, X_tests, y_train, y_tests, t_train, t_tests, train, tests = splits
    # y_train = [int(y[0]) for y in y_train]

    accs, nfrs, pfrs = [], [], []
    X_train_i = X_train
    y_train_i = y_train

    n_updates = len(X_tests) - 3
    for i in range(n_updates):
        print(f"> M{i} ({i}/{n_updates})")
        clf = LinearSVC()
        clf.fit(X_train_i, y_train_i)

        X_test_i = vstack((X_tests[i], X_tests[i+1]))
        y_test_i = np.hstack((y_tests[i], y_tests[i+1]))
        preds = clf.predict(X_test_i)
        precision_recall_fscore_support(y_test_i, preds, average='macro')
        # print(f"Acc = {acc*100} %")
        # accs.append(acc)

        preds1 = preds[:X_tests[i].shape[0]]

        if i > 0:
            nfr = compute_nflips(old_preds=preds2, new_preds=preds1) * 100
            pfr = compute_pflips(old_preds=preds2, new_preds=preds1) * 100
            print(f"NFR = {nfr} %")
            print(f"PFR = {pfr} %")
        else:
            nfr, pfr = None, None

        nfrs.append(nfr)
        pfrs.append(pfr)

        preds2 = preds[X_tests[i].shape[0]:]
        X_train_i = vstack((X_train_i, X_tests[i]))
        y_train_i = np.hstack((y_train_i, y_tests[i]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(accs, color='red', marker='o')
    ax[1].plot(nfrs, color='blue', marker='v')
    ax[2].plot(pfrs, color='green', marker='^')

    titles = ['Acc', 'NFR', 'PFR']
    for i, title in enumerate(titles):
        ax[i].set_title(title)
        ax[i].set_xlabel('Updates')
        ax[i].set_xticks(np.arange(n_updates))
    fig.legend()
    fig.tight_layout()
    fig.show()
    print("")






