from collections import defaultdict
import subprocess
import os.path

import numpy as np
import pandas as pd

from data import DATA_PATH

WORKERS = 8


class GaussianKernel:
    def __init__(self, sigma2=1.0):
        self.sigma2 = sigma2

    def __str__(self):
        return "Gaussian(sigma2={:.4f})".format(self.sigma2)

    def __call__(self, X1, X2):
        D = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        K = np.exp(-np.sum(D ** 2, axis=-1) / (2*self.sigma2))
        return K


class PolyKernel:
    def __init__(self, p=2, c=0.0, normalize=True):
        self.p = p
        self.c = c

    def __str__(self):
        return "PolyKernel(deg={p},c={c})".format(p=self.p, c=self.c)

    def __call__(self, X1, X2):
        K = (X1 @ X2.T + self.c) ** self.p
        Kxx1 = (np.sum(X1 * X1, axis=-1) + self.c) ** self.p
        Kxx2 = (np.sum(X2 * X2, axis=-1) + self.c) ** self.p
        norm = np.sqrt(Kxx1[:, np.newaxis]) * np.sqrt(Kxx2[np.newaxis, :])
        return K / norm


class SumKernel:
    def __init__(self, kernels, weights=None):
        self.kernels = kernels
        self.weights = weights if weights is not None else np.ones(
            len(kernels))

    def __str__(self):
        s = ""
        for K in self.kernels:
            s += str(K) + "+"
        return "SumKernel({})".format(s)

    def __call__(self, X1, X2, **kwargs):
        Ks = [K(X1, X2, **kwargs) for K in self.kernels]
        K = np.sum(self.weights[:, np.newaxis, np.newaxis]
                   * np.stack(Ks, axis=0), axis=0)
        return K


class SpectrumKernel:
    def __init__(self, k, normalize="full"):
        self.k = k
        self.normalize = normalize

    def __str__(self):
        return "Spectrum(k={})".format(self.k)

    def __call__(self, X1, X2, **kwargs):
        embd1 = [self._embed_string(s) for s in X1]
        embd2 = [self._embed_string(s) for s in X2]

        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                e1 = embd1[i]
                e2 = embd2[j]
                us = e1.keys() & e2.keys()
                for u in us:
                    K[i, j] += e1[u] * e2[u]

        if self.normalize in ["full", "sqrt"]:
            Kxx1 = np.array([self._norm(e) for e in embd1])
            Kxx2 = np.array([self._norm(e) for e in embd2])
            if self.normalize == "sqrt":
                Kxx1 = np.sqrt(Kxx1)
                Kxx2 = np.sqrt(Kxx2)
            norm = Kxx1[:, np.newaxis] * Kxx2[np.newaxis, :]
            K = K / norm

        return K

    def _embed_string(self, s):
        embd = defaultdict(int)
        for p in range(len(s) - self.k + 1):
            u = s[p:p+self.k]
            embd[u] += 1
        return embd

    def _norm(self, e):
        return sum([x ** 2 for x in e.values()])


class MismatchKernel(SpectrumKernel):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def __str__(self):
        return "Mismatch(k={k},alpha={alpha})".format(k=self.k, alpha=self.alpha)

    def _embed_string(self, s):
        embd = defaultdict(int)
        for p in range(len(s) - self.k + 1):
            u = s[p:p+self.k]
            embd[u] += 1
            for i in range(self.k):
                u1 = u[:i] + u[i+1:]
                embd[u1] += self.alpha
                for j in range(i+1, self.k):
                    u2 = u[:i] + u[i+1:j] + u[j+1:]
                    embd[u2] += self.alpha**2
        return embd


class SubstringKernel:
    def __init__(self, dataset_id, k, alpha, normalize=None):
        self.dataset_id = dataset_id
        self.k = k
        self.alpha = alpha
        self.normalize = normalize

    def __str__(self):
        return "Substring(k={k},alpha={alpha},norm={norm})".format(k=self.k, alpha=self.alpha, norm=self.normalize)

    def __call__(self, *args, test=False):
        if test:
            return FastSubstringKernelTest(d=self.dataset_id, k=self.k, alpha=self.alpha, normalize=self.normalize)()
        else:
            return FastSubstringKernel(d=self.dataset_id, k=self.k, alpha=self.alpha, normalize=self.normalize)()


class FastSubstringKernel:
    AUX_PATH = "kernels/{d}-subst-alpha={alpha}-{id}of{workers}-k="

    def __init__(self, d, k, alpha, normalize=None):
        self.d = d
        self.k = k
        self.alpha = alpha
        self.normalize = normalize
        self.in_path = DATA_PATH.format(tr_te='tr', k=d, mat='')

    def _split_work(self, N):
        total = (N*(N+1)/2) / WORKERS
        bounds = []
        p = 0
        for _ in range(WORKERS-1):
            q = int(N - np.sqrt(-2*total + N*N - 2*N*p + p*p))
            bounds.append((p, q))
            p = q
        bounds.append((p, N))
        return bounds

    def __call__(self, *args):
        os.makedirs(os.path.dirname(self.AUX_PATH), exist_ok=True)
        ps = []
        X = pd.read_csv(self.in_path)
        N = len(X)
        for id, (p, q) in enumerate(self._split_work(N)):
            aux_path = self.AUX_PATH.format(
                d=self.d, alpha=self.alpha, id=id, workers=WORKERS)
            out_path = aux_path.format(id=id) + str(self.k) + ".csv"
            if not os.path.isfile(out_path):
                ps.append(subprocess.Popen(
                    list(map(str, ["./substring", "train", self.in_path, aux_path, self.alpha, p, q]))))
        for p in ps:
            p.wait()
            if (p.returncode != 0):
                raise RuntimeError("The substring program crashed.")

        Ks = []
        for id, _ in enumerate(self._split_work(N)):
            aux_path = self.AUX_PATH.format(
                d=self.d, alpha=self.alpha, id=id, workers=WORKERS)
            out_path = aux_path.format(id) + str(self.k) + ".csv"
            Ks.append(np.loadtxt(out_path, delimiter=','))
        K = np.concatenate(Ks, axis=0)
        KT = np.copy(K.T)
        np.fill_diagonal(KT, 0)
        K = K + KT

        if self.normalize == 'sqrt':
            Kxx = np.sqrt(np.diagonal(K))
            norm = Kxx[:, np.newaxis] * Kxx[np.newaxis, :]
            K = K / norm
        elif self.normalize == 'mean':
            K = K / np.mean(K)

        return K


class FastSubstringKernelTest:
    AUX_PATH = "kernels/{d}-subst-test-alpha={alpha}-{id}of{workers}-k="

    def _split_work(self, N):
        step = N // WORKERS
        ps = [t * step for t in range(WORKERS)]
        qs = ps[1:] + [N]
        return list(zip(ps, qs))

    def __init__(self, d, k, alpha, normalize=None):
        self.d = d
        self.k = k
        self.alpha = alpha
        self.normalize = normalize
        self.in_path1 = DATA_PATH.format(tr_te='te', k=d, mat='')
        self.in_path2 = DATA_PATH.format(tr_te='tr', k=d, mat='')

    def __str__(self):
        return "Substring(k={k},alpha={alpha},norm={norm})".format(k=self.k, alpha=self.alpha, norm=self.normalize)

    def __call__(self, *args):
        os.makedirs(os.path.dirname(self.AUX_PATH), exist_ok=True)
        ps = []
        X = pd.read_csv(self.in_path1)
        N = len(X)
        for id, (p, q) in enumerate(self._split_work(N)):
            aux_path = self.AUX_PATH.format(
                d=self.d, alpha=self.alpha, id=id, workers=WORKERS)
            out_path = aux_path + str(self.k) + ".csv"
            if not os.path.isfile(out_path):
                ps.append(subprocess.Popen(
                    list(map(str, ['./substring', "test", self.in_path1, self.in_path2, aux_path, self.alpha, p, q]))))
        for p in ps:
            p.wait()

        Ks = []
        for id, _ in enumerate(self._split_work(N)):
            aux_path = self.AUX_PATH.format(
                d=self.d, alpha=self.alpha, id=id, workers=WORKERS)
            out_path = aux_path + str(self.k) + ".csv"
            Ks.append(np.loadtxt(out_path, delimiter=','))
        K = np.concatenate(Ks, axis=0)

        if self.normalize == "mean":
            K_train = FastSubstringKernel(
                self.d, k=self.k, alpha=self.alpha, normalize=None)
            K = K / np.mean(K_train)
        elif self.normalize == 'sqrt':
            Kxx_train, Kxx_test = FastSubstringKernelDiag(
                self.d, k=self.k, alpha=self.alpha)()
            norm = np.sqrt(Kxx_test[:, np.newaxis]) * \
                np.sqrt(Kxx_train[np.newaxis, :])
            K = K / norm

        return K


class FastSubstringKernelDiag:
    AUX_PATH = "kernels/{d}-subst-diag-alpha={alpha}-k="

    def __init__(self, d, k, alpha):
        self.d = d
        self.k = k
        self.alpha = alpha
        self.in_path = DATA_PATH.format(tr_te='te', k=d, mat='')

    def __call__(self, *args):
        os.makedirs(os.path.dirname(self.AUX_PATH), exist_ok=True)
        aux_path = self.AUX_PATH.format(d=self.d, alpha=self.alpha)
        out_path = aux_path + str(self.k) + ".csv"
        if not os.path.isfile(out_path):
            p = subprocess.Popen(
                list(map(str, ['./substring', "diag", self.in_path, aux_path, self.alpha])))
            p.wait()

        Kxx_test = np.loadtxt(out_path, delimiter=',')
        K_train = FastSubstringKernel(
            self.d, k=self.k, alpha=self.alpha, normalize=None)()
        Kxx_train = np.diag(K_train)
        return Kxx_train, Kxx_test
