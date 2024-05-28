import torch
import sklearn.metrics
import numpy as np
import random
import sys, string
import os, glob, re
from datetime import timedelta
import json, gc
import matplotlib.pyplot as plt
import matplotlib
from pandas import DatetimeIndex
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)
import torch
import torch.nn as nn


def correlation_coefficient_loss(outputs, targets):
    # Assuming outputs and targets are of shape [batch, features, time]
    mean_x = torch.mean(outputs, dim=2, keepdim=True)  # Mean across time
    mean_y = torch.mean(targets, dim=2, keepdim=True)
    vx = outputs - mean_x
    vy = targets - mean_y
    cost = torch.sum(vx * vy, dim=2) / (torch.sqrt(torch.sum(vx**2, dim=2)) *
                                        torch.sqrt(torch.sum(vy**2, dim=2)))
    return torch.mean(cost)  # Mean over batch


class CorrRMSELoss(nn.Module):

    def __init__(self, alpha=0.1):
        super(CorrRMSELoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        # RMSE calculation
        rmse = torch.sqrt(self.mse(outputs, targets))
        # Correlation calculation
        corr = correlation_coefficient_loss(outputs, targets)
        # Combine RMSE and correlation into a single loss
        return self.alpha * rmse - (1 - self.alpha) * corr


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})."
            )
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


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
