import logging
import json
import pickle
from datetime import datetime
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import os

DS_PATH = 'datasets/Android/features'

FEATURES = os.path.join(DS_PATH, 'drebin-parrot-v2-down-features-X.json')
LABELS = os.path.join(DS_PATH, 'drebin-parrot-v2-down-features-Y.json')
META = os.path.join(DS_PATH, 'drebin-parrot-v2-down-features-meta.json')
FAMILIES = os.path.join(DS_PATH, 'malware_families_meta.json')
VOCABULARY = os.path.join(DS_PATH, 'vocabulary.pkl')


def load_features(shas=False):
    logging.info('Loading features...')
    with open(FEATURES, 'rt') as f:
        X = json.load(f)
    if not shas:
        [o.pop('sha256') for o in X]

    with open(LABELS, 'rt') as f:
        y = json.load(f)

    with open(META, 'rt') as f:
        t = json.load(f)
    t = [o['dex_date'] for o in t]
    t = [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') for o in t]

    with open(FAMILIES, 'rt') as f:
        m = json.load(f)

    return X, y, t, m


def vectorize(X, y, t, m):
    """Transform input data into appropriate forms for an sklearn classifier.

    Args:
        X (list): A list of dictionaries of input features for each sample.
        y (list): A list of ground truths for the data.
        t (list): A list of datetimes for the data.

    """
    logging.info('Vectorizing features...')
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    with open(VOCABULARY, 'wb') as f:
        pickle.dump(vec, f)
    y = np.asarray(y)
    t = np.asarray(t)
    malware_families = [m[y_k.lower()] if y_l == '1' and y_k.lower() in m else None for y_l, y_k in y]
    malware_families = np.asarray(malware_families)
    return X, y, t, malware_families


def generate_random_temporal_features(n_samples_per_month=1000,
                                      unc_samples_per_month=0.1,
                                      n_features=1000,
                                      n_months=10):

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
    return X, y


if __name__ == "__main__":
    x, y, t, m = load_features()
    xv, yv, tv, mv = vectorize(x, y, t, m)

