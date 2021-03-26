import pickle
import numpy as np
import pandas as pd

from data import load_data
from train import kernels


def make_predictions(dataset_id, model=None, model_path=None):
    if model is None:
        if model_path is None:
            raise ValueError("Please supply either the model or its path.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    X_train, _ = load_data(dataset_id)
    X_test = load_data(dataset_id, test=True)

    kernel = kernels[dataset_id]
    K = kernel(X_test, X_train, test=True)

    y_te = model.predict(K) > 0

    y_te_df = pd.DataFrame(y_te, columns=['Bound'], dtype='int', index=pd.RangeIndex(
        start=1000*dataset_id, stop=1000*(dataset_id+1)))

    return y_te_df
