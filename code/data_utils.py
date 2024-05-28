from datetime import datetime, timedelta
import json
import string
from pandas import DatetimeIndex
import pandas as pd
import numpy as np
import json, gc
import torch
import os, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

from training_utils import *
import torch, os
from torch.multiprocessing import set_start_method
from datetime import datetime, timedelta

LR = 3e-06
device = "cuda" if torch.cuda.is_available() else "cpu"
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


def save_pkl(address, **dataframes):

    for suffix, df in dataframes.items():
        filename = f"{address}{suffix}.pkl"
        df.to_pickle(filename)


def load_pkl(address, *suffixes):
    dataframes = {}
    for suffix in suffixes:
        filename = f"{address}{suffix}.pkl"
        df = pd.read_pickle(filename)
        dataframes[suffix] = df
    return dataframes


def thousand_separator(x, pos):
    return f'{x:,.0f}'


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


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
    df[f"{column_name}_time_diff"] = df[column_name] - df[column_name].shift(
        days)
    df[f"{column_name}_time_diff"] = df[f"{column_name}_time_diff"].fillna(0.)
    df[column_name] = df[f"{column_name}_time_diff"]
    return df, initial_values


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
        input_cols, input_path):
    json_file_name = input_path + f"/x_data_aux/experiment_ids.json"
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
