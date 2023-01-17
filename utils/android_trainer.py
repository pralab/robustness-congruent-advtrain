from sklearn.svm import LinearSVC
import numpy as np
import os
from utils.load_android_features import DS_PATH, generate_random_temporal_features
import pickle
from tesseract import temporal
from utils.eval import compute_nflips, compute_pflips, compute_common_nflips
from utils.android_eval import compute_all_metrics, adjust_threshold, compute_churn_metrics
from utils.data import ds_stack, ds_unstack
from scipy.sparse import vstack, csr_matrix
from collections import OrderedDict

class AndroidTemporalTrainer:
    def __init__(self, results_path,
                 train_size,
                 test_size,
                 val_size=0,
                 fpr=0.01,
                 n_updates=None,
                 C=1,
                 class_weight='balanced',
                 sample_weight=None,
                 temporal_weight=False,
                 max_iter=1000,
                 overwrite=False
                 ):
        keys = list(locals().keys())
        keys = keys[keys.index('self') + 1:]

        hp_dict = {}
        for key in keys:
            hp_dict[key] = eval(key)
        self.hp_dict = hp_dict

        self.results_path = results_path
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.fpr = fpr
        self.n_updates = n_updates
        self.C = C
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.temporal_weight = temporal_weight
        self.max_iter = max_iter
        self.overwrite = overwrite

        self._clf = None
        self.clf_sequence = OrderedDict()

        self.metrics = {}


    def load_splitted_android_dataset(self, train_size=0, test_size=1, granularity='month'):
        with open(os.path.join(DS_PATH, 'drebin_xyt.pkl'), 'rb') as f:
            ds = pickle.load(f)
        X, y, t, m = ds['X'], ds['y'], ds['t'], ds['m']
        y = np.array([int(y[0]) for y in y])
        # Partition dataset
        # qui prendo il dataset splittato in mesi, senza fare ancora train-test split
        _, X, _, y, *_ = temporal.time_aware_train_test_split(
            X, y, t, train_size=train_size,
            test_size=test_size,
            granularity=granularity)

        # self.X, self.y = X, y

        return X, y

    def reset_classifier(self):
        self._clf = LinearSVC(C=self.C,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter)

    def adjust_clf_threshold(self, X_val, y_val):
        threshold = adjust_threshold(self._clf, X_val, y_val, self.fpr)

        self._clf.intercept_[0] = self._clf.intercept_[0] - threshold

        # import matplotlib.pyplot as plt
        # plt.plot(fpr, tpr)
        # plt.axvline(x=fpr[idx])
        # plt.show()
        return threshold

    def reset_metrics(self):
        self.metrics = {}

    def update_metrics(self, metrics):
        # todo: qui ci posso mettere qualcosa come args*
        # self.metrics['tprs'].append(tpr * 100)
        # self.metrics['fprs'].append(fpr * 100)
        # self.metrics['precs'].append(prec * 100)
        # self.metrics['recs'].append(rec * 100)
        # self.metrics['f1s'].append(f1 * 100)

        for metric_name, value in metrics.items():
            # value = value*100 if value is not None else value
            if metric_name not in self.metrics.keys():
                self.metrics[metric_name] = [value]
            else:
                self.metrics[metric_name].append(value)


    def train_sequence_parametric(self):
        X, y = self.load_splitted_android_dataset()
        #todo: capire come rendere più parametrico per scorrere su altri parametri
        results = []
        for row, sample_weight_k in enumerate(self.sample_weight):
            t = f" sample_weight={sample_weight_k} "
            print(f"{t:#^40}")

            self.train_sequence(X, y, sample_weight=sample_weight_k)
            self.evaluate_sequence(X, y)

            self.hp_dict['sample_weight'] = sample_weight_k
            result = {**self.metrics, **self.hp_dict}
            results.append(result)
            self.reset_metrics()

        with open(os.path.join(self.results_path), 'wb') as f:
            pickle.dump(results, f)

    def train_sequence(self, X, y,
                       sample_weight=None):

        # If n_updates is not given compute maximum updates available
        if self.n_updates is None:
            self.n_updates = len(X) - self.train_size - self.test_size


        # Iterate over the updates
        for i in range(self.n_updates):
            print(f"\n> M{i}/{self.n_updates}")

            ############################
            # DATA
            ############################

            # todo: fare un interfaccia comoda per lo scorrere dei mesi nel train, val e test
            # Obtain train window
            X_train_i, y_train_i, train_idxs = ds_stack(X, y,
                                                        start=i,
                                                        n_months=self.train_size - self.val_size)
            if self.val_size > 0:
                # Obtain validation window
                X_val_i, y_val_i, val_idxs = ds_stack(X, y,
                                                      start=i + self.train_size - self.val_size,
                                                      n_months=self.val_size)

            ############################
            # TRAIN
            ############################

            # Churn-aware filter: it computes sample weights only if a previous classifier exists
            sample_weights = self.churn_aware_filter(sample_weight, X_train_i, y_train_i)

            print(f"Train months: {len(train_idxs)}, N samples: {X_train_i.shape[0]}")
            # todo: parametric clf type
            self.reset_classifier() #reset self._clf
            self._clf.fit(X_train_i,
                          y_train_i,
                          sample_weight=sample_weights
                          )

            ############################
            # SET THRESHOLD WITH VALIDATION
            ############################

            threshold = 0  # default threshold at 0
            if self.val_size > 0:
                # Adjust threshold by checking FPR on the validation set
                self.adjust_clf_threshold(X_val_i, y_val_i) #fpr <= 1% by default

            self.clf_sequence[i] = self._clf    # save new clf update

        return

    def evaluate_sequence(self, X, y):
        assert len(self.clf_sequence) > 0, \
            "You must run train_sequence first!"

        for i, clf_i in self.clf_sequence.items():
            # Here I have to perform 2 evaluations:
            # - Mi on Ti
            # - Mi-1 on Ti (to correlate with NFR calculation)

            ############################
            # EVALUATION
            ############################

            # Obtain test window
            X_test_i, y_test_i, test_idxs = ds_stack(X, y,
                                                     start=self.train_size + i,
                                                     n_months=self.test_size)
            print(f"Test months: {len(test_idxs)}, N samples: {X_test_i.shape[0]}")

            preds = clf_i.predict(X_test_i)
            performance_metrics = compute_all_metrics(preds, y_test_i)

            if i > 0:
                # If i>0 I can compute NFR wrt to previous clf
                clf_i_prev = self.clf_sequence[i-1]     # Pick the previous clf
                old_preds = clf_i_prev.predict(X_test_i)    # evaluate the previous clf on the current test set
                old_performance_metrics = compute_all_metrics(old_preds, y_test_i)
            else:
                old_preds = None
                old_performance_metrics = None


            performance_metrics['old_perf'] = old_performance_metrics
            churn_metrics = compute_churn_metrics(preds, old_preds, y_test_i)

            metrics_i = {**performance_metrics, **churn_metrics}

            # # todo: trovare modo comodo per tutte ste variabili delle metriche
            # # append results in each metric
            self.update_metrics(metrics_i)

        old_perf = {}
        old_perf_backup = self.metrics.pop('old_perf')
        for key in old_perf_backup[1].keys():
            old_key = f"old_{key}"
            old_perf[old_key] = []
            for perf in old_perf_backup:
                perf = perf[key] if perf is not None else perf
                old_perf[old_key].append(perf)

        self.metrics = {**self.metrics, **old_perf}

        print("")
            # result = {
            #     'f1s': f1s,
            #     'precs': precs,
            #     'recs': recs,
            #     'tprs': tprs,
            #     'fprs': fprs,
            #     'nfrs_pos': nfrs_pos,
            #     'nfrs_neg': nfrs_neg,
            #     'pfrs_pos': pfrs_pos,
            #     'pfrs_neg': pfrs_neg,
            #     'nfrs_tot': nfrs_tot,
            #     'pfrs_tot': pfrs_tot,
            #     'class_weight': self.class_weight,
            #     'sample_weights': sample_weight
            # }


    def churn_aware_filter(self, sample_weight, X_train_i, y_train_i):
        if (sample_weight is not None) and (self._clf is not None):
            # This clf is the one at the iteration i-1!!!
            preds_tr = self._clf.predict(X_train_i)
            sample_weights = np.ones(preds_tr.shape)
            # Set higher weights for sample predicted correctly by clf i-1
            sample_weights[preds_tr == y_train_i] = sample_weight
        else:
            sample_weights = np.ones(X_train_i.shape[0])

        if self.temporal_weight:
            temporal_weights = np.linspace(0.1, 1, num=sample_weights.shape[0])
            sample_weights = sample_weights * temporal_weights

        return sample_weights

    def get_hyperparameters_dict(self, keys):
        hp_dict = {}

        for key in keys:
            pass