import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from data_utils import *
from training_utils import *

matplotlib.use("Agg")
import os
from datetime import datetime

from data_formatters import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error

current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
START, END = datetime(2020, 3, 1), datetime(2022, 3, 1)


def timestamp_to_unix(timestamp):
    return timestamp.value // 10**9


def serialize_dict(data):
    if isinstance(data, dict):
        return {k: serialize_dict(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.float64, np.int32)):
        return int(data) if isinstance(data, np.int64) else float(data)
    elif isinstance(data, DatetimeIndex):
        return data.strftime("%Y-%m-%d").tolist()
    return data


class PoissonHurdleModel:

    def __init__(self):
        self.zero_model = LogisticRegression()
        self.count_model = PoissonRegressor()

    def fit(self, X, y):
        print("X, y", X, y)
        try:

            y_binary = (y > 0).astype(int)
            print("y_binary", y_binary)

            if len(np.unique(y_binary)) > 1:
                self.zero_model.fit(X, y_binary)
                X_non_zero = X[y > 0]
                y_non_zero = y[y > 0]
                self.count_model.fit(X_non_zero, y_non_zero)
            else:
                print(
                    "Only one class in y_binary, skipping zero_model fitting.")
                self.count_model.fit(X, y)
        except Exception as e:
            print(f"An error occurred: {e}")

            self.count_model.fit(X, y)

    def predict(self, X):
        try:

            if hasattr(self.zero_model, "classes_") and len(
                    self.zero_model.classes_) > 1:
                prob_zero = self.zero_model.predict_proba(X)[:, 0]
                count_pred = self.count_model.predict(X)
                return (1 - prob_zero) * count_pred
            else:

                count_pred = self.count_model.predict(X)
                return count_pred
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return self.count_model.predict(X)

    def get_rmse(self, true_values, predicted_values):

        filtered_errors = [(true, pred)
                           for true, pred in zip(true_values, predicted_values)
                           if pred <= true]

        if not filtered_errors:
            return 0.0

        true_filtered, pred_filtered = zip(*filtered_errors)
        return np.sqrt(mean_squared_error(true_filtered, pred_filtered))


class PoissonHurdleModel_orig:

    def __init__(self):
        self.zero_model = LogisticRegression()
        self.count_model = PoissonRegressor()

    def fit(self, X, y):
        print("X, y", X, y)

        y_binary = (y > 0).astype(int)
        print("y_binary", y_binary)
        self.zero_model.fit(X, y_binary)

        X_non_zero = X[y > 0]
        y_non_zero = y[y > 0]
        self.count_model.fit(X_non_zero, y_non_zero)

    def predict(self, X):
        prob_zero = self.zero_model.predict_proba(X)[:, 0]
        count_pred = self.count_model.predict(X)
        return prob_zero * 0 + (1 - prob_zero) * count_pred

    def get_rmse(self, true_values, predicted_values):
        return np.sqrt(mean_squared_error(true_values, predicted_values))
