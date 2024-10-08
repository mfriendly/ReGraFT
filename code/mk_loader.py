import json
import os
import types

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

from data_formatters import ts_dataset as ts_dataset
from data_utils import *
from training_utils import *


def Prepare_Training_Inputs(config, args, list_of_states, list_of_abbr, states,
                            device):

    args.PREFIX = f"{args.MODEL}_{args.past_steps}_{args.future_steps}"
    args.PREFIX_fuzzy = f"*{args.MODEL}*{args.past_steps}_{args.future_steps}"
    config.update({"PREFIX": args.PREFIX, "PREFIX_fuzzy": args.PREFIX_fuzzy})
    set_seed(args.SEED)

    config["skip_connection"] = True

    config["add_target_to_scale_vars"] = True

    config["save_transform_weight"] = False
    config["START"], config["END"] = "2020-03-01", "2022-03-01"
    config["dropout_type"] = "zoneout"
    config["fusion_mode"] = "none"

    config["gcn_alpha"] = 1.0

    config["difference-Days"] = 1

    config["apply-difference-to-X"] = True

    config["save_path"] = os.path.join(args.output_path, args.data_name,
                                       args.nation, args.PREFIX, args.INPUTID,
                                       str(args.SEED))

    config = set_save_paths(config, args)
    try:
        (
            train_dataset,
            val_dataset,
            test_dataset,
            train,
            valid,
            test,
            config,
            input_cols,
            target_col,
            holiday_cols,
            INPUTID,
            standard_scaler_stats_dict,
            initial_values_date_dict,
            const_dict,
        ) = make_loaders(config, args, device)
    except Exception as e:
        import traceback

        print(os.path.basename(__file__), e)
        traceback.print_exc()
        (
            train_dataset,
            val_dataset,
            test_dataset,
            train,
            valid,
            test,
            config,
            input_cols,
            target_col,
            holiday_cols,
            INPUTID,
            standard_scaler_stats_dict,
            initial_values_date_dict,
            const_dict,
        ) = make_loaders(config, args, device)

    train_tmp, valid_tmp, test_tmp = train, valid, test

    config["activation"] = "selu"

    config, args = retrieve_last_processed_give_lc_path(config, args)
    R = f"{config['save_path']}/config.json"
    save_to_json(config, R)
    args = types.SimpleNamespace(**config)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train,
        valid,
        test,
        config,
        input_cols,
        target_col,
        holiday_cols,
        INPUTID,
        standard_scaler_stats_dict,
        initial_values_date_dict,
    )


def make_loaders(config, args, device):

    nation = args.nation
    data_name = args.data_name

    embedding_sizes = None
    standard_scaler_stats_dict, initial_values_date_dict = [None] * 2
    const_dict = {}
    scaler_name = config["scaler_name"]
    print("scaler_name")

    embedding_dim = config["embedding_dim"]
    SPLIT = config["SPLIT"]
    data_name = config["data_name"]

    PREFIX = config["PREFIX"]

    INPUTID = config["INPUTID"]

    (list_of_states, list_of_abbr, states2abbr,
     adjs) = retrieve_metadata_of_nation(config, args.nation, args.input_path,
                                         args.device)

    train_data1 = {}
    valid_data = {}
    test_data = {}
    save_data = config[
        "input_path"] + f"/x_data_pkl/{nation}/{INPUTID}/{scaler_name}"
    if not os.path.exists(save_data):
        os.makedirs(save_data)

    config = update_inputs_config(config)
    config["rnn_layers"] = 1

    max_samples_list = []
    path_check = os.path.join(save_data, "test.pt")
    if config["CONTINUE"] and os.path.exists(path_check):
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
        id_col, static_cols, time_col, max_samples, num_static, output_size, train, PREFIX, test = [
            None
        ] * 9
        headers = ["train", "val", "test"]
        train = torch.load(os.path.join(save_data, headers[0] + ".pt"),
                           map_location="cpu")
        valid = torch.load(os.path.join(save_data, headers[1] + ".pt"),
                           map_location="cpu")
        test = torch.load(os.path.join(save_data, headers[2] + ".pt"),
                          map_location="cpu")

        config["id_col"] = id_col
        config["static_cols"] = static_cols
        config["time_col"] = time_col

        config["max_samples"] = max_samples

        config["num_static"] = config["static_variables_len"]
        config["output_size"] = 1

        config["data_name"] = data_name
        config["scaler_name"] = scaler_name
        config["nation"] = nation

        train_dataset = ts_dataset.TSDataset(config, mode="train", data=train)
        val_dataset = ts_dataset.TSDataset(config, mode="val", data=valid)
        test_dataset = ts_dataset.TSDataset(config, mode="test", data=test)

    else:
        standard_scaler_stats_dict = {}
        initial_values_date_dict = {ab: None for ab in list_of_abbr}
        for node_id, abbr in enumerate(list_of_abbr):
            print("node_id, abbr", node_id, abbr)
            dfname = config[
                "input_path"] + f"/x_data_df/{nation}/{str(node_id).zfill(2)}_{abbr}.csv"
            print("dfname", dfname)
            if not os.path.exists(dfname):
                continue
            df = pd.read_csv(dfname)
            df["new_confirmed"] = df["new_confirmed"].abs()
            df["date"] = pd.to_datetime(df["date"])
            df["new_confirmed"] = df["new_confirmed"].mask(
                df["new_confirmed"] == 0.0, np.nan)
            df["new_confirmed"] = (df["new_confirmed"].interpolate(
                method="cubic").fillna(method="bfill").fillna(method="ffill"))
            df["new_confirmed"] = (df["new_confirmed"].fillna(
                method="ffill").fillna(method="bfill").fillna(0).astype(float))
            df["new_confirmed"] = df["new_confirmed"].abs()
            constant = df["new_confirmed"].min()
            const_dict[abbr] = constant
            df_final = interpolate_data(df, abbr)
            cols_numerical_to_process = time_varying_real_variables_encoder = config[
                "unknown_real_x_target"]
            df = smoothing(df_final, cols_numerical_to_process)
            if not config["apply-difference-to-X"]:
                df["new_confirmed_Y"] = df["new_confirmed"]
                DiffCol = "new_confirmed_Y"
            else:
                DiffCol = "new_confirmed"
            if config["diff"]:
                df, initial_values = difference_from_previous_time(
                    df,
                    DiffCol,
                    list_of_abbr,
                    abbr,
                    days=config["difference-Days"])
                initial_values_date_dict[abbr] = initial_values
            if config["apply-difference-to-X"]:
                df["new_confirmed_Y"] = df["new_confirmed"]
            else:
                pass

            def retrieve_category_embedding_sizes(
                    config, time_varying_categorical_variables):
                embedding_sizes = {}
                for col in time_varying_categorical_variables:
                    df[col] = df[col].astype("int64").fillna(int(99))
                    if col not in df.columns:
                        df[col] = 0
                    df[col] = df[col].astype("int").astype("str").astype(
                        "category")
                    num = df[col].nunique() + 1
                    embedding_sizes[col] = (int(num), int(embedding_dim))

                config["x_categorical_embed_sizes"] = embedding_sizes
                return config

            config = update_inputs_config(config)

            time_varying_categorical_variables = config[
                "time_varying_categorical_variables_list"]
            config = retrieve_category_embedding_sizes(
                config, time_varying_categorical_variables)
            df = df[["date"] + config["input_cols"]]
            df = df.reset_index()
            valid_boundary = int(len(df) * SPLIT[0])
            test_boundary = int(len(df) * SPLIT[1])
            df_unscaled = df.copy()
            test_unscaled = test_unscaled_diff = df_unscaled[df_unscaled.index
                                                             >= test_boundary]
            if config["add_target_to_scale_vars"]:
                cols_numerical_to_process = ["new_confirmed_Y"
                                             ] + cols_numerical_to_process
            else:
                cols_numerical_to_process = cols_numerical_to_process
            if config["scaler_name"] == "":
                scaler_standard = StandardScaler()
                df.loc[:test_boundary, [
                    f"{col}" for col in cols_numerical_to_process
                ]] = scaler_standard.fit_transform(
                    df.loc[:test_boundary,
                           [f"{col}" for col in cols_numerical_to_process]])
                df.loc[test_boundary:,
                       [f"{col}" for col in cols_numerical_to_process
                        ]] = scaler_standard.transform(df[[
                            f"{col}" for col in cols_numerical_to_process
                        ]][test_boundary:])
                standard_scaler_stats_dict[abbr] = {
                    "mean": scaler_standard.mean_[0],
                    "std": scaler_standard.scale_[0]
                }
            else:
                standard_scaler_stats_dict[abbr] = {"mean": None, "std": None}
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
        input_cols = config.get("input_cols", 99)
        train = pd.concat([t for t in train_data1.values()],
                          axis=0)[["date"] + input_cols]
        valid = pd.concat([t for t in valid_data.values()],
                          axis=0)[["date"] + input_cols]
        test = pd.concat([t for t in test_data.values()],
                         axis=0)[["date"] + input_cols]

        lll = [train, valid, test]
        for idf, df in enumerate(lll):
            df = df[input_cols + ["date"]]
            unique_dates_count = df.date.nunique()
            max_samples = unique_dates_count
            max_samples_list.append(max_samples)

        if scaler_name != "" and scaler_name not in INPUTID:
            INPUTID += f"-{scaler_name}"
        config["INPUTID"] = INPUTID
        headers = ["train", "val", "test"]
        torch.save(train, os.path.join(save_data, headers[0] + ".pt"))
        torch.save(valid, os.path.join(save_data, headers[1] + ".pt"))
        torch.save(test, os.path.join(save_data, headers[2] + ".pt"))

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
        max_samples_list
        config["id_col"] = input_cols[-1]

        config["static_cols"] = config["static_variables_len"]
        config = update_inputs_config(config)
        config["max_samples"] = max_samples

        config["output_size"] = 1

        config["data_name"] = data_name
        config["scaler_name"] = scaler_name
        config["nation"] = nation

        train_dataset = ts_dataset.TSDataset(config, mode="train", data=train)
        val_dataset = ts_dataset.TSDataset(config, mode="val", data=valid)
        test_dataset = ts_dataset.TSDataset(config, mode="test", data=test)

    config["max_samples_list"] = max_samples_list
    args = types.SimpleNamespace(**config)
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train,
        valid,
        test,
        config,
        args.input_cols,
        args.target_col,
        args.holiday_cols,
        args.INPUTID,
        standard_scaler_stats_dict,
        initial_values_date_dict,
        const_dict,
    )
