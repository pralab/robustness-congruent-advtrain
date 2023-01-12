from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH, generate_random_temporal_features
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips
from utils.data import ds_stack, ds_unstack
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score


def train_sequence_svm(results_path, train_size, test_size, n_updates=None,
                       C=1,
                       class_weight='balanced',
                       sample_weight_list=None,
                       temporal_weight=False,
                       max_iter=1000,
                       overwrite=False
                       ):

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
    for row, sample_weight in enumerate(sample_weight_list):
        t = f" sample_weight={sample_weight} "
        print(f"{t:#^40}")

        # X_train_i = X[0]
        # y_train_i = y[0]
        # X_tests, y_tests = X[1:], y[1:]

        precs, recs, f1s = [], [], []
        nfrs_pos, pfrs_pos = [], []
        nfrs_neg, pfrs_neg = [], []

        if n_updates is None:
            n_updates = len(X) - train_size - test_size

        sample_weights = None
        for i in range(n_updates):
            print(f"\n> M{i}/{n_updates}")

            # Obtain train window
            X_train_i, y_train_i, train_idxs = ds_stack(X, y,
                                                        start=i,
                                                        n_months=train_size)


            # Churn-aware filter
            if (sample_weight is not None) and (i > 0):
                # must be float or integer
                preds_tr = clf.predict(X_train_i)
                sample_weights = np.ones(preds_tr.shape)
                sample_weights[preds_tr == y_train_i] = sample_weight
                # sample_weights[train_idxs[-2]:] = 1
            else:
                sample_weights = np.ones(X_train_i.shape[0])

            if temporal_weight:
                temporal_weights = np.linspace(0.1, 1, num=sample_weights.shape[0])
                sample_weights = sample_weights * temporal_weights

            print(f"Train months: {len(train_idxs)}, N samples: {X_train_i.shape[0]}")
            clf = LinearSVC(C=C,
                            class_weight=class_weight,
                            max_iter=max_iter)
            clf.fit(X_train_i,
                    y_train_i,
                    sample_weight=sample_weights
                    )

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
                'class_weight': class_weight,
                'sample_weights': sample_weight
            }
        )

    with open(os.path.join(results_path), 'wb') as f:
        pickle.dump(results, f)