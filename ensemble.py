import numpy as np


class Ensemble:
    def __init__(self, models, pis):
        self.models = models
        self.pis = pis

    def predict(self, K):
        N = K.shape[0]
        M = len(self.models)
        row_idx = np.arange(N)[:, np.newaxis]
        preds = np.zeros((N, M))
        for i in range(M):
            col_idx = self.pis[i][np.newaxis, :]
            K_test = K[row_idx, col_idx]
            preds[:, i] = self.models[i].predict(K_test)
        pred = np.sum(preds, axis=1)
        return np.sign(pred)
