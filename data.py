import numpy as np
import pandas as pd

DATA_PATH = "data/X{tr_te}{k}{mat}.csv"
LABELS_PATH = "data/Ytr{k}.csv"


def train_test_split(X, y, train_ratio, pi=None):
    N = len(X)
    n_train = int(N * train_ratio)
    if pi is None:
        pi = np.random.permutation(N)
    y_train = y[pi[:n_train]]
    y_test = y[pi[n_train:]]
    if isinstance(X, np.ndarray):
        X_train = X[pi[:n_train]]
        X_test = X[pi[n_train:]]
    else:
        X_train = list(map(X.__getitem__, pi[:n_train]))
        X_test = list(map(X.__getitem__, pi[n_train:]))
    return X_train, X_test, y_train, y_test


def kernel_train_test_split(K, y, train_ratio, pi=None):
    N = len(K)
    n_train = int(N * train_ratio)
    if pi is None:
        pi = np.random.permutation(N)
    y_train = y[pi[:n_train]]
    y_test = y[pi[n_train:]]

    row_idx = pi[:n_train, np.newaxis]
    col_idx = pi[np.newaxis, :n_train]
    K_train = K[row_idx, col_idx]

    row_idx = pi[n_train:, np.newaxis]
    K_test = K[row_idx, col_idx]

    return K_train, K_test, y_train, y_test


def load_data(k, embedded=False, test=False):
    tr_te = "te" if test else "tr"
    if embedded:
        X_path = DATA_PATH.format(k=k, tr_te=tr_te, mat="_mat100")
        X = pd.read_csv(X_path, sep=' ', header=None).to_numpy()
    else:
        X_path = DATA_PATH.format(k=k, tr_te=tr_te, mat="")
        X = pd.read_csv(X_path)['seq'].tolist()

    if test:
        return X

    y_path = LABELS_PATH.format(k=k)
    y = pd.read_csv(y_path)['Bound'].to_numpy()
    y = 2*y - 1
    return X, y
