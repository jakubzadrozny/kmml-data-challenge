import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import load_data, kernel_train_test_split
from ridge import KRRClassifier
from logistic import KernelLogisticClassifier
from svm import KSVM
from utils import KernelCrossValidation
from train import select_model
from predict import make_predictions

RESULT_PATH = "data/Yte.csv"

if __name__ == '__main__':
    dfs = []
    for dataset_id in range(3):

        print("Processing dataset", dataset_id)

        _, model = select_model(dataset_id)
        df = make_predictions(dataset_id, model)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(RESULT_PATH, index_label="Id")
