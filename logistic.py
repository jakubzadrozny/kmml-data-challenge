import numpy as np

from model import Model
from ridge import WRR, KWRR


class LogisticClassifier(Model):
    def __init__(self, lbd, eps=1e-4, n_iter=30):
        self.lbd = lbd
        self.eps = eps
        self.n_iter = n_iter

    def __str__(self):
        return "Logistic"

    def fit(self, X, y):
        X = self._pad_with_ones(X)
        theta = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            m = X @ theta
            w = self._sigmoid(m) * self._sigmoid(-m)
            z = m + y / self._sigmoid(y * m)
            model = WRR(lbd=self.lbd)
            model.fit(X, z, w)
            if self._norm(theta - model.w) < self.eps:
                break
            theta = model.w
        self.w = theta

    def predict(self, X):
        X = self._pad_with_ones(X)
        pred = X @ self.w
        return np.sign(pred)


class KernelLogisticClassifier(Model):
    def __init__(self, lbd, eps=1e-4, n_iter=30):
        self.lbd = lbd
        self.eps = eps
        self.n_iter = n_iter

    def __str__(self):
        return "KernelLogistic"

    def fit(self, K, y):
        alpha = np.zeros(K.shape[0])
        for _ in range(self.n_iter):
            m = K @ alpha
            w = self._sigmoid(m) * self._sigmoid(-m)
            z = m + y / self._sigmoid(y * m)
            model = KWRR(lbd=self.lbd)
            model.fit(K, z, w)
            if self._norm(alpha - model.w) < self.eps:
                break
            alpha = model.w
        self.w = alpha

    def predict(self, K):
        pred = K @ self.w
        return np.sign(pred)
