import numpy as np

from model import Model


class WRR(Model):
    def __init__(self, lbd):
        self.lbd = lbd

    def fit(self, X, y, w):
        X_T_w = X.T @ np.diag(w)
        self.w = np.linalg.solve(
            X_T_w @ X + self.lbd * X.shape[0] * np.eye(X.shape[1]), X_T_w @ y)

    def predict(self, X):
        return X @ self.w


class RRClassifier(Model):
    def __init__(self, lbd):
        self.wrr = WRR(lbd)

    def __str__(self):
        return "RR"

    def fit(self, X, y):
        X = self._pad_with_ones(X)
        w = np.ones(X.shape[0])
        self.wrr.fit(X, y, w)

    def predict(self, X):
        X = self._pad_with_ones(X)
        pred = self.wrr.predict(X)
        return np.sign(pred)


class KWRR(Model):
    def __init__(self, lbd):
        self.lbd = lbd

    def fit(self, K, y, w):
        w_inv = np.diag(1 / w)
        try:
            self.w = np.linalg.solve(K + self.lbd * K.shape[0] * w_inv, y)
        except np.linalg.LinAlgError:
            w_half = np.diag(np.sqrt(w))
            self.w = w_half @ np.linalg.inv(
                w_half @ K @ w_half + self.lbd * K.shape[0] * np.eye(K.shape[0])) @ w_half @ y

    def predict(self, K):
        # K is the kernel matrix between X_test and X_train
        return K @ self.w


class KRRClassifier(Model):
    def __init__(self, lbd):
        self.kwrr = KWRR(lbd)

    def __str__(self):
        return "KRR"

    def fit(self, K, y):
        w = np.ones(K.shape[0])
        self.kwrr.fit(K, y, w)

    def predict(self, K):
        pred = self.kwrr.predict(K)
        return np.sign(pred)
