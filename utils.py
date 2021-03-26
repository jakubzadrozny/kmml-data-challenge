from multiprocessing import Pool
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import kernel_train_test_split
from ensemble import Ensemble


def accuracy(y_pred, y_real):
    return np.sum(y_pred == y_real) / len(y_real)


def prec_recall_f1(y_pred, y_real):
    y_pred_bool = y_pred > 0
    y_real_bool = y_real > 0
    tp = np.sum(np.logical_and(y_pred_bool, y_real_bool))
    prec = 0 if tp == 0 else tp / np.sum(y_pred_bool)
    recall = 0 if tp == 0 else tp / np.sum(y_real_bool)
    f1 = 0 if prec + recall == 0 else 2 * prec * recall / (prec + recall)
    return prec, recall, f1


class KernelCrossValidation:
    def __init__(self, models, kernels, lbds, model_path=None, results_path=None,
                 criterion='accuracy', folds=5, ensemble_size=251, train_ratio=0.8, n_jobs=0):
        self.models = models
        self.lbds = lbds
        self.criterion = criterion
        self.folds = folds
        self.train_ratio = train_ratio
        self.kernels = kernels
        self.model_path = model_path
        self.results_path = results_path
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs

    def _fit_per_kernel(self, kernel_idx):
        kernel = self.kernels[kernel_idx]
        K = kernel(self.X, self.X)

        params = []
        results = []
        # for f in tqdm(range(self.folds)):
        for f in range(self.folds):
            K_train, K_val, y_train, y_val = kernel_train_test_split(
                K, self.y, self.train_ratio, pi=self.pis[f])

            rows_results = []
            for model_idx, model_class in enumerate(self.models):
                for lbd in self.lbds:
                    model = model_class(lbd=lbd)
                    try:
                        model.fit(K_train, y_train)
                        y_pred = model.predict(K_val)
                    except:
                        print("ERROR", kernel, model)
                        y_pred = np.zeros_like(y_val)

                    prec, recall, f1 = prec_recall_f1(y_pred, y_val)
                    rows_results.append({
                        'accuracy': accuracy(y_pred, y_val),
                        'precision': prec,
                        'recall': recall,
                        'f1': f1,
                    })

                    if f == 0:
                        params.append({
                            'model_idx': model_idx,
                            'kernel_idx': kernel_idx,
                            'lambda': lbd,
                        })

            results_df = pd.DataFrame(rows_results)
            results.append(results_df)

        params = pd.DataFrame(params)
        results = pd.concat(results)
        results = results.groupby(results.index).mean()
        results = pd.concat((params, results), axis='columns')
        return results

    def fit(self, X, y):
        print("Fitting models...")

        self.X = X
        self.y = y

        self.pis = [np.random.permutation(len(X))
                    for _ in range(self.folds)]

        results = []
        if self.n_jobs > 0:
            pool = Pool(self.n_jobs)
            for df in tqdm(pool.imap_unordered(self._fit_per_kernel, range(len(self.kernels))), total=len(self.kernels)):
                results.append(df)

            pool.close()
            pool.join()
        else:
            for df in tqdm(map(self._fit_per_kernel, range(len(self.kernels))), total=len(self.kernels)):
                results.append(df)

        results = pd.concat(results, ignore_index=True)

        results['model'] = results.apply(lambda x: str(
            self.models[int(x['model_idx'])]), axis='columns')
        results['kernel'] = results.apply(lambda x: str(
            self.kernels[int(x['kernel_idx'])]), axis='columns')

        idx_best = results[self.criterion].idxmax()
        best_model_class = self.models[results.loc[idx_best, 'model_idx']]
        best_kernel = self.kernels[results.loc[idx_best, 'kernel_idx']]
        results = results.drop(columns=['model_idx', 'kernel_idx'])
        self.best_params = results.loc[idx_best]

        print('Validation done.')
        print('Retraining the best model...')
        print(self.best_params)

        K = best_kernel(X, X)
        pis = [np.random.permutation(len(X))
               for _ in range(self.ensemble_size)]
        rows = []
        models = []
        for f in tqdm(range(self.ensemble_size)):
            K_train, K_test, y_train, y_test = kernel_train_test_split(
                K, y, self.train_ratio, pi=pis[f])
            model = best_model_class(lbd=self.best_params["lambda"])
            model.fit(K_train, y_train)
            y_pred = model.predict(K_test)
            prec, recall, f1 = prec_recall_f1(y_pred, y_test)
            rows.append({
                "acc": accuracy(y_pred, y_test),
                "precision": prec,
                "recall": recall,
                "f1": f1,
            })
            models.append(model)
        test_results = pd.DataFrame(rows).mean(axis=0)
        test_std = pd.DataFrame(rows).std(axis=0)
        print("Final scores:")
        print(test_results)
        print("with std")
        print(test_std)

        if self.results_path is not None:
            results.sort_values(by=self.criterion, ascending=False).to_csv(
                self.results_path, index=False)
        self.results = results

        n_train = int(K.shape[0] * self.train_ratio)
        pis_train = [pis[f][:n_train] for f in range(self.ensemble_size)]
        self.best_model = Ensemble(models=models, pis=pis_train)

        if self.model_path is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.best_model, f)

        return self.results, self.best_model
