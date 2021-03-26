import numpy as np
from cvxopt import matrix, solvers

from model import Model

solvers.options['show_progress'] = False


class KSVM(Model):
    def __init__(self, lbd):
        self.lbd = lbd

    def __str__(self):
        return "SVM"

    def fit(self, K, y):
        G2 = np.diag(y)
        h2 = np.full_like(y, 1/(2*self.lbd*y.shape[0]))
        G1 = -G2
        h1 = np.zeros_like(y)
        G = np.concatenate((G1, G2), axis=0)
        h = np.concatenate((h1, h2), axis=0)

        sol = solvers.qp(matrix(K, tc='d'),
                         matrix(-y, tc='d'),
                         matrix(G, tc='d'),
                         matrix(h, tc='d'))
        self.w = np.array(sol['x']).flatten()

    def predict(self, K):
        pred = K @ self.w
        return np.sign(pred)
