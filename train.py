import numpy as np

from data import load_data
from ridge import KRRClassifier
from logistic import KernelLogisticClassifier
from svm import KSVM
from utils import KernelCrossValidation
from kernels import SpectrumKernel, SumKernel, SubstringKernel

SEED = 47
FOLDS = 10

model_path = None  # set this to save trained model in a file
results_path = None  # set this to save CV results in a .csv file

kernels = [
    # Kernel for D0
    [
        SumKernel([
            SubstringKernel(dataset_id=0, k=10, alpha=0.23, normalize="sqrt"),
            SpectrumKernel(k=9, normalize="sqrt"),
        ]),
    ],

    # Kernel for D1
    [
        SumKernel([
            SubstringKernel(dataset_id=1, k=9, alpha=0.27, normalize="sqrt"),
            SubstringKernel(dataset_id=1, k=10, alpha=0.27, normalize="sqrt"),
            SubstringKernel(dataset_id=1, k=8, alpha=0.23, normalize="sqrt"),
            SpectrumKernel(k=8, normalize="sqrt"),
            SpectrumKernel(k=6, normalize="sqrt"),
            SpectrumKernel(k=5, normalize="sqrt"),
        ]),
    ],

    # Kernel for D2
    [
        SumKernel([
            SubstringKernel(dataset_id=2, k=7, alpha=0.27, normalize="sqrt"),
            SubstringKernel(dataset_id=2, k=8, alpha=0.25, normalize="sqrt"),
            SpectrumKernel(k=7, normalize="sqrt"),
            SpectrumKernel(k=6, normalize="sqrt"),
        ]),
    ],
]

lambdas = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    5e-4,
    1e-3,
    3e-3,
    5e-3,
    1e-2,
]

models = [
    KRRClassifier,
    KernelLogisticClassifier,
    KSVM,
]


def select_model(dataset_id):
    np.random.seed(SEED)
    X, y = load_data(dataset_id)
    cv = KernelCrossValidation(models, kernels[dataset_id], lambdas,
                               folds=FOLDS, model_path=model_path, results_path=results_path)
    return cv.fit(X, y)


if __name__ == '__main__':
    select_model(0)
