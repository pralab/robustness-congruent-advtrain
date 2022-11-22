from sklearn.svm import LinearSVC
import numpy as np
#from utils.load_features import load_features, vectorize
#from tesseract import temporal
from utils.eval import compute_nflips

import matplotlib.pyplot as plt

if __name__ == "__main__":
    #    X, y, t, m = load_features()
    #    X, y, t, m = vectorize(X, y, t, m)
    # # Partition dataset
    #    splits = temporal.time_aware_train_test_split(
    #        X, y, t, train_size=12, test_size=1, granularity='quartal')
    #    X_actual, X_tests, y_actual, y_tests, t_actual, t_tests, train, tests = splits

    n_samples_per_month = 1000
    unc_samples_per_month = 0.1
    n_features = 1000

    n_months = 1
    X, y = [], []
    n = []
    for t in range(n_months):
        delta = n_samples_per_month * unc_samples_per_month
        delta_n_samples = np.random.randint(
            int(delta*2)) - int(delta)
        n_samples = n_samples_per_month + delta_n_samples
        # n_samples will deviate from n_samples_per_month by +/- (n_samples_per_month * unc_samples_per_month)

        X_t = np.random.randint(2, size=(n_samples, n_features))
        y_t = np.random.randint(2, size=n_samples)
        X.append(X_t)
        y.append(y_t)

    clfs = []
    for t in range(len(X)):
        clf = LinearSVC()
        clf.fit(X[t], y[t])
        preds = clf.predict(X[t + 1])

        acc = (preds == y[t + 1]).mean()
        print(acc)

    print("")






