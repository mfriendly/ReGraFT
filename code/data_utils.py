from datetime import datetime, timedelta
import json
import string
from pandas import DatetimeIndex
import pandas as pd
import numpy as np
import json, gc
import torch
import os, pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm
import seaborn as sns
#from pylab import rcParams
import matplotlib

matplotlib.use("Agg")
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from datetime import timedelta

LR = 3e-06
from training_utils import *
import torch, os
from torch.multiprocessing import set_start_method

device = "cuda"
torch.set_default_device(device)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
sys.setrecursionlimit(50000)
script_dir = ""
if script_dir not in sys.path:
    sys.path.append(script_dir)
script_name = os.path.basename(__file__).replace(".", "_")
try:
    set_start_method("spawn")
except RuntimeError:
    pass
import warnings

warnings.filterwarnings("ignore")


def thousand_separator(x, pos):
    return f'{x:,.0f}'


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


from datetime import datetime, timedelta


def reconstruct_series_to_array(differences,
                                initial_values,
                                start_date_,
                                end_date_,
                                date_format="%Y-%m-%d",
                                abbr=None,
                                days=1,
                                future_steps=42):
    start_date = datetime.strptime(start_date_, date_format) if isinstance(
        start_date_, str) else start_date_
    start_date = start_date + timedelta(days=future_steps)
    end_date = datetime.strptime(end_date_, date_format) if isinstance(
        end_date_, str) else end_date_
    dates = pd.date_range(start_date, end_date)
    reconstructed_values = [
        initial_values[(start_date -
                        timedelta(days=days)).strftime(date_format)]
        for start_date in dates
    ]
    for i in range(len(differences)):
        new_value = reconstructed_values[i] + differences[i][0]
        reconstructed_values[i] = new_value
    reconstructed_values = np.array(reconstructed_values).reshape(-1, 1)
    return reconstructed_values


def difference_from_previous_time(df,
                                  column_name="new_confirmed_Y",
                                  list_of_abbr=None,
                                  abbr=None,
                                  days=1):
    date_column_name = 'date'
    initial_values = df[[date_column_name, column_name
                         ]].set_index(date_column_name)[column_name].to_dict()
    start_date = datetime.strptime('2021-09-12', '%Y-%m-%d')
    end_date = datetime.strptime('2022-03-01', '%Y-%m-%d')
    initial_values = {
        key.strftime('%Y-%m-%d'): value
        for key, value in initial_values.items()
        if start_date <= key <= end_date
    }
    df[f"{column_name}_week_diff"] = df[column_name] - df[column_name].shift(
        days)
    df[f"{column_name}_week_diff"] = df[f"{column_name}_week_diff"].fillna(0.)
    df[column_name] = df[f"{column_name}_week_diff"]
    return df, initial_values


def adjust_outliers(df, state, epidemic_column="new_confirmed"):
    state_indices = df[df["State"] == state].index
    roll_df = df.loc[state_indices,
                     epidemic_column].rolling(7, min_periods=1).mean()
    negative_or_zero_indices = df.loc[state_indices][df.loc[
        state_indices, epidemic_column] <= 0].index
    df.loc[negative_or_zero_indices,
           epidemic_column] = roll_df.loc[negative_or_zero_indices]
    df.loc[state_indices,
           epidemic_column] = (df.loc[state_indices, epidemic_column].fillna(
               method="ffill").fillna(method="bfill").fillna(0.0))
    return df


def interpolate_data(df, state, epidemic_column="new_confirmed"):
    state_df = df[df["State"] == state].copy()
    for i, row in state_df.iterrows():
        if (i % 7) != 0:
            state_df.at[i, epidemic_column] = np.nan
    state_df[epidemic_column] = state_df[epidemic_column].interpolate(
        method="linear")
    df.loc[state_df.index, epidemic_column] = state_df[epidemic_column]
    return df


def replace_cols(df, dict_):
    df.rename(columns=dict_)
    return df


def get_values_in_date_range(csv_path,
                             start_date,
                             end_date,
                             value_column="new_confirmed"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered_df = df.loc[mask]
    values = filtered_df["new_confirmed"]
    values_array = values.to_numpy()
    return values_array


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


def format_date(unix_timestamp):
    if isinstance(unix_timestamp, int):
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
        except Exception as e:
            print(e)
            pass
            print(e)
            return None
    elif isinstance(unix_timestamp, datetime.datetime):
        dt = unix_timestamp
    else:
        return None
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"


def filter_dates(df, start_date, end_date, abbr):
    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame(date_range, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df_full = pd.merge(date_df, df, on="date", how="left")
    for col in df_full.columns:
        if pd.api.types.is_numeric_dtype(df_full[col]):
            df_full[col] = df_full[col].fillna(method="ffill").fillna(0)
        elif pd.api.types.is_datetime64_any_dtype(
                df_full[col]) or pd.api.types.is_categorical_dtype(
                    df_full[col]):
            df_full[col] = df_full[col].fillna(method="ffill").fillna(
                method="bfill")
        elif df_full[col].dtype == "object":
            df_full[col] = df_full[col].fillna(method="ffill").fillna(
                method="bfill")
        else:
            df_full[col] = df_full[col].fillna(method="ffill")
    return df_full


def retrive_config_by_INPUTID_and_dir(config_json_path):
    with open(config_json_path, "r") as json_file:
        config = json.load(json_file)
    return config


def retrieve_experiment_id(input_cols):
    json_file_name = ".." + f"/data/x_data_aux/experiment_ids.json"
    with open(json_file_name, "r") as json_file:
        experiment_ids = json.load(json_file)
    subset_key = "_".join(sorted(input_cols))
    experiment_id = experiment_ids.get(subset_key, "ID not found")
    return experiment_id


def make_regraft_dataset(df):
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["categorical_month"] = df["date"].dt.month
    df["days_from_start"] = (df["date"] - df["date"].min()).dt.days
    df["week_of_month"] = df["day"] // 7 + 1
    df["categorical_month"] = df["categorical_month"].astype("category")
    df["categorical_day_of_week"] = df["day_of_week"].astype("category")
    df["categorical_week"] = df["week_of_month"].astype("category")
    df["new_confirmed"] = df["new_confirmed"].interpolate(
        method='cubic').fillna(method="bfill").fillna(method="ffill")
    return df


def invert_difference(history, column_name, yhat, interval=1):
    if interval == 0:
        return yhat
    else:
        last_value = history[column_name].iloc[-interval]
        original_scale_value = yhat + last_value
        return original_scale_value


def smoothing(df, real_columns):
    df1 = df.copy()
    df0 = df1.iloc[:6]
    df0.loc[:6, real_columns] = np.nan
    df1 = df1.iloc[6:]
    for col in real_columns:
        temp = []
        for i in range(6, len(df)):
            ave = np.mean(df[col].iloc[i - 6:i + 1])
            temp.append(ave)
        df1[col] = temp
        try:
            df1.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            pass
        try:
            df1.drop(["level_0"], axis=1, inplace=True)
        except:
            pass
    df2 = pd.concat([df0, df1], axis=0)
    df2 = df2.bfill()
    return df2


def save_pickle(file, filename):
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file


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


def format_date(unix_timestamp):
    if isinstance(unix_timestamp, int):
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
        except Exception as e:
            print(e)
            pass
            return None
    elif isinstance(unix_timestamp, datetime.datetime):
        dt = unix_timestamp
    else:
        return None
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"


def filter_dates(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame(date_range, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df_full = pd.merge(date_df, df, on="date", how="left")
    try:
        df_full = df_full.fillna(method="ffill").fillna(0.0).astype(float)
    except Exception as e:
        print(e)
    return df_full


def standardize_numeric_cols_except_target(train_df, valid_df, test_df,
                                           numeric_cols):
    mean = train_df[numeric_cols].mean()
    std = train_df[numeric_cols].std()
    train_df_std = train_df.copy()
    valid_df_std = valid_df.copy()
    test_df_std = test_df.copy()
    train_df_std[numeric_cols] = (train_df[numeric_cols] - mean) / std
    valid_df_std[numeric_cols] = (valid_df[numeric_cols] - mean) / std
    test_df_std[numeric_cols] = (test_df[numeric_cols] - mean) / std
    return train_df_std, valid_df_std, test_df_std


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


def save_to_json(dict_, name):
    with open(name, "w") as f:
        json.dump(serialize_dict(dict_), f, indent=4)


def generate_next_identifier(current_index):
    base = len(string.ascii_uppercase)
    result = ""
    while True:
        quotient, remainder = divmod(current_index, base)
        result = string.ascii_uppercase[remainder] + result
        if quotient == 0:
            break
        current_index = quotient - 1
    return result


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
        next_identifier = generate_next_identifier(len(experiment_ids))
        new_experiment_id = f"{len(input_cols)}{next_identifier}"
        experiment_ids[subset_key] = new_experiment_id
        with open(json_file_name, "w") as json_file:
            json.dump(experiment_ids, json_file, indent=4)
        return new_experiment_id
