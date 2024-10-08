import json
import pickle
import string
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import torch
from pandas import DatetimeIndex

matplotlib.use("Agg")
import warnings
from datetime import datetime, timedelta

import torch

warnings.filterwarnings("ignore")


def save_pkl(address, **dataframes_or_tensors):
    for suffix, data in dataframes_or_tensors.items():
        filename = f"{address}{suffix}.pkl"
        if isinstance(data, pd.DataFrame):
            data.to_pickle(filename)
        elif isinstance(data, torch.Tensor):
            torch.save(data, filename)


def load_pkl(device, address, *suffixes):
    dataframes_or_tensors = {}
    for suffix in suffixes:
        filename = f"{address}{suffix}.pkl"
        try:
            data = pd.read_pickle(filename)
        except (ValueError, IOError):
            data = torch.load(filename, map_location=device)
        dataframes_or_tensors[suffix] = data
    return dataframes_or_tensors


def thousand_separator(x, pos):
    return f"{x:, .0f}"


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
    date_column_name = "date"
    initial_values = df[[date_column_name, column_name
                         ]].set_index(date_column_name)[column_name].to_dict()
    start_date = datetime.strptime("2021-09-12", "%Y-%m-%d")
    end_date = datetime.strptime("2022-03-01", "%Y-%m-%d")
    initial_values = {
        key.strftime("%Y-%m-%d"): value
        for key, value in initial_values.items()
        if start_date <= key <= end_date
    }
    df[f"{column_name}_time_diff"] = df[column_name] - df[column_name].shift(
        days)
    df[f"{column_name}_time_diff"] = df[f"{column_name}_time_diff"].fillna(0.0)
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


import pickle

import torch


def save_pickle(file, filename):
    if isinstance(file, torch.Tensor):
        torch.save(file, filename)
    else:
        with open(filename, "wb") as f:
            pickle.dump(file, f)


import io
import pickle

import torch


def load_pickle(device, filename):
    try:
        print(
            f"Trying to load {filename} as a torch object on device {device}")
        file = torch.load(filename, map_location=device)
        print(f"Successfully loaded {filename} as a torch object")
    except (pickle.UnpicklingError, ValueError, IOError, RuntimeError) as e:
        print(f"Failed to load {filename} as a torch object: {e}")
        try:

            with open(filename, "rb") as f:
                buffer = io.BytesIO(f.read())
            file = torch.load(buffer, map_location=device)
            print(
                f"Successfully loaded {filename} from buffer as a torch object"
            )
        except Exception as e:
            print(
                f"Failed to load {filename} from buffer as a torch object: {e}"
            )
            try:

                with open(filename, "rb") as f:
                    file = pickle.load(f)
                print(f"Successfully loaded {filename} as a pickle object")
            except Exception as e:
                print(f"Failed to load {filename} as a pickle object: {e}")
                raise e
    return file


def format_date(unix_timestamp):
    if isinstance(unix_timestamp, int):
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
        except Exception as e:
            print(e)
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
        (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64),
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
