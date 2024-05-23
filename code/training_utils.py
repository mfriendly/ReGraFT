import torch
import sklearn.metrics
import numpy as np
import random
import sys, string
import os, glob, re
from datetime import timedelta
import json, gc
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
from pandas import DatetimeIndex
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)
import time, json


def check_if_exp_id_exists_from_reading_json_then_create_one_if_not(
        input_cols):
    json_file_name = ".." + f"/data/x_data_aux/experiment_ids.json"
    try:
        with open(json_file_name, "r") as json_file:
            experiment_ids = json.load(json_file)
    except FileNotFoundError:
        experiment_ids = {}
    subset_key = "_".join(sorted(input_cols))
    if subset_key in experiment_ids:
        return experiment_ids[subset_key]
    else:
        next_identifier_index = len(experiment_ids) % len(
            string.ascii_uppercase)
        next_identifier = string.ascii_uppercase[next_identifier_index]
        new_experiment_id = f"{len(input_cols)}{next_identifier}"
        experiment_ids[subset_key] = new_experiment_id
        with open(json_file_name, "w") as json_file:
            json.dump(experiment_ids, json_file, indent=4)
        return new_experiment_id


def set_seed(seed=42, make_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_last_processed_number(save_path, number):
    with open(save_path + "/last_processed.txt", "w") as file:
        file.write(str(number))
        import re


def get_last_processed_number(path):
    files = glob.glob(os.path.join(path, "*.pth"))
    max_int = -1
    for file_path in files:
        base_name = os.path.basename(file_path)
        numbers = [int(num) for num in re.findall(r"\d+", base_name)]
        if numbers:
            max_int = max(max_int, max(numbers))
    return max_int if max_int != -1 else 0


def RMSE_(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())


def MAE(predictions, targets):
    r = np.abs(predictions - targets).mean()
    return r


def MAPE_(predictions, targets):
    return np.mean(np.abs((targets - predictions) / targets)) * 100


def CORR_(true, pred):
    true = np.atleast_1d(true)
    pred = np.atleast_1d(pred)
    true_mean = true.mean() if true.size > 0 else 0
    pred_mean = pred.mean() if pred.size > 0 else 0
    u = ((true - true_mean) * (pred - pred_mean)).sum()
    d = np.sqrt(((true - true_mean)**2).sum() * ((pred - pred_mean)**2).sum())
    if d == 0:
        return 0
    corr = u / d
    return corr if np.isscalar(corr) else corr.mean() if corr.size > 0 else 0


def window_evaluation(config, label, outputs, label_unscaled,
                      outputs_unscaled):
    future_steps = config['future_steps']
    label, outputs = label.reshape(1, -1), outputs.reshape(1, -1)
    windows = [7, 14, 21, 28, 35, 42]
    metrics = [
        [RMSE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
        [MAE(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
        [MAPE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
        [CORR_(label[:, i - 7:i], outputs[:, i - 7:i]) * 100 for i in windows],
    ]
    metrics
    return metrics


def set_seed(seed=42, make_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def serialize_dict(data):
    if isinstance(data, dict):
        return {k: serialize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, np.ndarray):
        return serialize_dict(data.tolist())
    elif isinstance(
            data,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, DatetimeIndex):
        return data.strftime("%Y-%m-%d").tolist()
    return data
