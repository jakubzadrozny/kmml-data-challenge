import numpy as np


class Model:
    def __init__(self):
        pass

    def _pad_with_ones(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _norm(self, x):
        return np.sqrt(np.sum(x ** 2))
