import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import json, gc, os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns

sns.set_style("whitegrid")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import torch, os
import csv
from data_formatters import ts_dataset as ts_dataset
from training_utils import *
from data_utils import *
from datetime import datetime
import glob
import json
import string

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def make_loaders(config,
                 input_path,
                 SEED,
                 CONTINUE,
                 data_name,
                 diff=False,
                 pct=False,
                 aug=False,
                 scaler_name=""):
    PLOT = False
    embedding_sizes = None
    standard_scaler_stats_dict, initial_values_date_dict = [None] * 2
    const_dict = {}
    scaler_name = config["scaler_name"]
    num_nodes = 51
    BATCHSIZE = 4
    past_steps = 42
    future_steps = 42
    output_size = 1
    unknown_real_x_target = config["unknown_real_x_target"]
    input_cols = config["input_cols"]
    holiday_cols = config["holiday_cols"]
    try:
        policy_lag_vars = config["policy_lag_vars"]
        gt_lag_vars = config["gt_lag_vars"]
        time_cat_cols = config["time_cat_cols"]
    except:
        pass
    target_col = config["target_col"]
    EMBEDDING_DIM = config["EMBEDDING_DIM"]
    SPLIT = config["SPLIT"]
    data_name = config["data_name"]
    try:
        PREFIX = config["PREFIX"]
    except:
        pass
    INPUTID = config["INPUTID"]
    stateUS = pd.read_csv(input_path + f"/x_data_aux/statemappingUS.csv")
    list_of_states = stateUS["State"].tolist()
    list_of_abbr = stateUS["Abbr"].tolist()
    dict_of_states = dict(zip(list_of_states, list_of_abbr))
    SS = input_path + f"/x_data_aux/dist_matrix_US.pkl"
    DD = input_path + f"/x_data_aux/dir_travel_matrix_US.pkl"
    if not os.path.exists(DD):
        PATH = input_path + f"/x_data_aux"
        travel_matrix = np.load(PATH + f"/matrix_1.npy")
        save_pickle(travel_matrix, DD)
        dist_matrix = np.load(PATH + f"/matrix_0.npy")
        travel_matrix = np.load(PATH + f"/matrix_1.npy")
        save_pickle(dist_matrix, SS)
    else:

        travel_matrix = load_pickle(DD)
        dist_matrix = load_pickle(SS)
    train_data1 = {}
    train_data2 = {}
    valid_data = {}
    test_data = {}
    save_data = input_path + f"/x_data_pkl/{INPUTID}/{scaler_name}"
    if not os.path.exists(save_data):
        os.makedirs(save_data)
    mode = 'val'
    file_paths = {
        "inputs": f"{save_data}/{mode}_inputs.pkl",
        "outputs": f"{save_data}/{mode}_outputs.pkl",
        "metadata": f"{save_data}/{mode}_metadata.pkl",
    }
    input_cols = config["input_cols"]
    target_col = input_cols[0]
    time_col = "days_from_start"
    static_cols = []
    max_samples_list = []
    id_col = input_cols[-2]
    input_size = len(input_cols) - 2
    config["input_size"] = input_size
    config["static_variables"] = 0
    time_varying_real_variables_encoder = (unknown_real_x_target +
                                           policy_lag_vars + gt_lag_vars)
    time_varying_categorical_variables = holiday_cols + time_cat_cols
    x_categoricals = time_varying_categorical_variables
    config["x_categoricals"] = x_categoricals
    config["time_varying_categorical_variables"] = len(
        time_varying_categorical_variables)
    config["time_varying_real_variables_encoder"] = len(
        time_varying_real_variables_encoder)
    config[
        "time_varying_categorical_variables_list"] = time_varying_categorical_variables
    config[
        "time_varying_real_variables_encoder_list"] = time_varying_real_variables_encoder
    config["input_cols"] = input_cols
    config["time_varying_real_variables_decoder"] = (
        len(time_varying_real_variables_encoder) - 1)
    config["embedding_dim"] = EMBEDDING_DIM
    rnn_hidden_dimension = config["embedding_dim"] * (
        len(time_varying_real_variables_encoder) +
        len(time_varying_categorical_variables))
    config["rnn_hidden_dimension"] = rnn_hidden_dimension
    config["num_masked_series"] = 1
    config["rnn_layers"] = 1
    config["device"] = "cuda"
    config["encode_length"] = past_steps
    config["seq_length"] = total_steps = past_steps + future_steps
    Rt = f"{save_data}initial_values_date_dict.json"
    if CONTINUE:
        alist = [
            "embedding_sizes", "standard_scaler_stats_dict",
            "initial_values_date_dict", "const_dict"
        ]
        blist = [
            embedding_sizes, standard_scaler_stats_dict,
            initial_values_date_dict, const_dict
        ]
        try:
            results = []
            for filename in alist:
                R = os.path.join(save_data, f"{filename}.json")
                with open(R, "r") as json_file:
                    data = json.load(json_file)
                    results.append(data)
            (embedding_sizes, standard_scaler_stats_dict,
             initial_values_date_dict, const_dict) = results
            config["x_categorical_embed_sizes"] = embedding_sizes
        except Exception as e:
            print(e, "line 142")
            pass
        id_col, static_cols, time_col, max_samples, num_static, output_size, train, PREFIX, test = [
            None
        ] * 9
        dataframes = load_pkl(save_data, 'train', 'valid', 'test')
        train = dataframes['train']
        valid = dataframes['valid']
        test = dataframes['test']
        train_dataset = ts_dataset.TSDataset(
            CONTINUE=CONTINUE,
            mode="train",
            id_col=id_col,
            static_cols=static_cols,
            time_col=time_col,
            input_cols=input_cols,
            target_col=target_col,
            total_steps=total_steps,
            max_samples=max_samples,
            input_size=input_size,
            past_steps=past_steps,
            num_static=num_static,
            output_size=output_size,
            data=train,
            num_nodes=num_nodes,
            future_steps=future_steps,
            prefix=PREFIX,
            INPUT_ID=INPUTID,
            data_name=data_name,
            scaler=scaler_name,
        )
        val_dataset = ts_dataset.TSDataset(
            CONTINUE=CONTINUE,
            mode='val',
            id_col=id_col,
            static_cols=static_cols,
            time_col=time_col,
            input_cols=input_cols,
            target_col=target_col,
            total_steps=total_steps,
            max_samples=max_samples,
            input_size=input_size,
            past_steps=past_steps,
            num_static=num_static,
            output_size=output_size,
            data=valid,
            num_nodes=num_nodes,
            future_steps=future_steps,
            prefix=PREFIX,
            INPUT_ID=INPUTID,
            data_name=data_name,
            scaler=scaler_name,
        )
        test_dataset = ts_dataset.TSDataset(
            CONTINUE=CONTINUE,
            mode="test",
            id_col=id_col,
            static_cols=static_cols,
            time_col=time_col,
            input_cols=input_cols,
            target_col=target_col,
            total_steps=total_steps,
            max_samples=max_samples,
            input_size=input_size,
            past_steps=past_steps,
            num_static=num_static,
            output_size=output_size,
            data=None,
            num_nodes=num_nodes,
            future_steps=future_steps,
            prefix=PREFIX,
            INPUT_ID=INPUTID,
            data_name=data_name,
            scaler=scaler_name,
        )
    else:
        standard_scaler_stats_dict = {}
        initial_values_date_dict = {ab: None for ab in list_of_abbr}
        for node_id, abbr in enumerate(list_of_abbr):
            if abbr == "0.0":
                continue
            dfname = input_path + f"/x_data_df/df_{abbr}.csv"
            if not os.path.exists(dfname):
                continue
            df = pd.read_csv(dfname)
            df['new_confirmed'] = df['new_confirmed'].abs()
            df['date'] = pd.to_datetime(df['date'])
            df["new_confirmed"] = df["new_confirmed"].mask(
                df["new_confirmed"] == 0., np.nan)
            df["new_confirmed"] = df["new_confirmed"].interpolate(
                method='cubic').fillna(method="bfill").fillna(method="ffill")
            df['new_confirmed'] = df['new_confirmed'].fillna(
                method="ffill").fillna(method="bfill").fillna(0).astype(float)
            df['new_confirmed'] = df['new_confirmed'].abs()
            constant = df['new_confirmed'].min()
            const_dict[abbr] = constant
            df_final = interpolate_data(df, abbr)
            cols_numerical_to_process = time_varying_real_variables_encoder
            df = smoothing(df_final, cols_numerical_to_process)
            if not config['apply-difference-to-X']:
                df["new_confirmed_Y"] = df["new_confirmed"]
                DiffCol = "new_confirmed_Y"
            else:
                DiffCol = "new_confirmed"
            if diff:
                df, initial_values = difference_from_previous_time(
                    df,
                    DiffCol,
                    list_of_abbr,
                    abbr,
                    days=config['difference-Days'])
                initial_values_date_dict[abbr] = initial_values
            if config['apply-difference-to-X']:
                df["new_confirmed_Y"] = df["new_confirmed"]
            else:
                pass
            embedding_sizes = {}
            for col in time_varying_categorical_variables:
                df[col] = df[col].astype("int64").fillna(int(99))
                if col not in df.columns:
                    df[col] = 0
                df[col] = df[col].astype("int").astype("str").astype(
                    "category")
                num = df[col].nunique() + 1
                embedding_sizes[col] = (int(num), int(EMBEDDING_DIM))
            df = df[["date"] + input_cols]
            config["x_categorical_embed_sizes"] = embedding_sizes
            df = df.reset_index()
            valid_boundary = int(len(df) * SPLIT[0])
            test_boundary = int(len(df) * SPLIT[1])
            df_unscaled = df.copy()
            test_unscaled = test_unscaled_diff = df_unscaled[df_unscaled.index
                                                             >= test_boundary]
            if config["add_target_to_scale_vars"]:
                cols_numerical_to_process = ['new_confirmed_Y'
                                             ] + cols_numerical_to_process
            else:
                cols_numerical_to_process = cols_numerical_to_process
            scaler_standard = StandardScaler()
            df.loc[:test_boundary,
                   [f"{col}" for col in cols_numerical_to_process
                    ]] = scaler_standard.fit_transform(
                        df.loc[:test_boundary,
                               [f"{col}"
                                for col in cols_numerical_to_process]])
            df.loc[test_boundary:,
                   [f"{col}" for col in cols_numerical_to_process
                    ]] = scaler_standard.transform(df[[
                        f"{col}" for col in cols_numerical_to_process
                    ]][test_boundary:])
            standard_scaler_stats_dict[abbr] = {
                "mean": scaler_standard.mean_[0],
                "std": scaler_standard.scale_[0]
            }
            df.reset_index(drop=True, inplace=True)
            train = df[df.index < valid_boundary]
            valid = df[(df.index >= valid_boundary)
                       & (df.index < test_boundary)]
            test = df[df.index >= test_boundary]
            for dataset in [train, valid, test]:
                dataset.reset_index(drop=True, inplace=True)
            train_data1[abbr] = train
            valid_data[abbr] = valid
            test_data[abbr] = test
        train = pd.concat([t for t in train_data1.values()],
                          axis=0)[["date"] + input_cols]
        valid = pd.concat([t for t in valid_data.values()],
                          axis=0)[["date"] + input_cols]
        test = pd.concat([t for t in test_data.values()],
                         axis=0)[["date"] + input_cols]
        num_static = 1
        PREFIX = config["PREFIX"]
        config["PREFIX"] = PREFIX
        lll = [train, valid, test]
        for idf, df in enumerate(lll):
            df = df[input_cols + ["date"]]
            unique_dates_count = df.date.nunique()
            max_samples = unique_dates_count
            max_samples_list.append(max_samples)
        headers = ["train", "val", "test"]
        if scaler_name != "" and scaler_name not in INPUTID:
            INPUTID += f"-{scaler_name}"
        config["INPUTID"] = INPUTID
        save_pkl(save_data, train=train, valid=valid, test=test)
        alist = [
            "embedding_sizes", "standard_scaler_stats_dict",
            "initial_values_date_dict", "const_dict"
        ]
        blist = [
            embedding_sizes, standard_scaler_stats_dict,
            initial_values_date_dict, const_dict
        ]
        for aa, bb in zip(alist, blist):
            R = f"{save_data}/{aa}.json"
            save_to_json(bb, R)
        PREFIX = None
        max_samples_list
        train_dataset = ts_dataset.TSDataset(
            CONTINUE=CONTINUE,
            mode="train",
            id_col=id_col,
            static_cols=static_cols,
            time_col=time_col,
            input_cols=input_cols,
            target_col=target_col,
            total_steps=total_steps,
            max_samples=max_samples_list[0],
            input_size=input_size,
            past_steps=past_steps,
            num_static=num_static,
            output_size=output_size,
            data=train,
            num_nodes=num_nodes,
            future_steps=future_steps,
            prefix=PREFIX,
            INPUT_ID=INPUTID,
            data_name=data_name,
            scaler=scaler_name,
        )
        val_dataset = ts_dataset.TSDataset(CONTINUE=CONTINUE,
                                           mode='val',
                                           id_col=id_col,
                                           static_cols=static_cols,
                                           time_col=time_col,
                                           input_cols=input_cols,
                                           target_col=target_col,
                                           total_steps=total_steps,
                                           max_samples=max_samples_list[1],
                                           input_size=input_size,
                                           past_steps=past_steps,
                                           num_static=num_static,
                                           output_size=output_size,
                                           data=valid,
                                           num_nodes=num_nodes,
                                           future_steps=future_steps,
                                           prefix=PREFIX,
                                           INPUT_ID=INPUTID,
                                           data_name=data_name,
                                           scaler=scaler_name)
        test_dataset = ts_dataset.TSDataset(
            CONTINUE=CONTINUE,
            mode="test",
            id_col=id_col,
            static_cols=static_cols,
            time_col=time_col,
            input_cols=input_cols,
            target_col=target_col,
            total_steps=total_steps,
            max_samples=max_samples_list[2],
            input_size=input_size,
            past_steps=past_steps,
            num_static=num_static,
            output_size=output_size,
            data=test,
            num_nodes=num_nodes,
            future_steps=future_steps,
            prefix=PREFIX,
            INPUT_ID=INPUTID,
            data_name=data_name,
            scaler=scaler_name,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCHSIZE,
        num_workers=0,
        shuffle=True,
        generator=torch.Generator(device=device),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCHSIZE,
        num_workers=0,
        shuffle=False,
        generator=torch.Generator(device=device),
        drop_last=True,
    )
    test_loader = None
    standard_scaler_stats_dict_one = None
    minmax_scalers_dict_one = None
    config["max_samples_list"] = max_samples_list
    return (train, valid, test, train_loader, val_loader, test_loader, config,
            dist_matrix, travel_matrix, input_cols, target_col,
            policy_lag_vars, holiday_cols, INPUTID, standard_scaler_stats_dict,
            standard_scaler_stats_dict_one, test_dataset,
            initial_values_date_dict, const_dict)
