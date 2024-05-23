import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import json, gc
from torch.utils.data import Subset
from datetime import timedelta

LR = 3e-06

from data_utils import *
from training_utils import *
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import torch, os
import csv

import importlib, sys

import torch.optim as optim
from datetime import datetime
import glob
from torch.multiprocessing import set_start_method
from tqdm import tqdm
import json
import string

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


def Test(config, model_, test_loader, adjs, scaler_name,
         standard_scaler_stats_dict, diff, initial_values_date_dict):

    model_.eval()
    save_path = config['save_path']
    INPUTID = config['INPUTID']
    SEED = config['SEED']
    MODEL = config["MODEL"]
    stateUS = pd.read_csv(".." + f"/data/x_data_aux/statemappingUS.csv")
    list_of_states = stateUS["State"].tolist()
    list_of_abbr = stateUS["Abbr"].tolist()
    past_steps = config['past_steps']
    attention_dict = {
        ab: {
            "encoder_attn_AVG": None,
            "decoder_attn_AVG": None,
            "encoder_attn_SUM": np.zeros((past_steps, config["input_size"])),
            "decoder_attn_SUM": np.zeros((past_steps, config["input_size"]))
        }
        for ab in list_of_abbr
    }
    transform_dict = {
        index: {
            "transform_ws_AVG": None,
            "transform_ws_SUM": np.zeros((past_steps, past_steps))
        }
        for index in range(4)
    }
    adjs_dict = {
        "encoder_adjs_AVG": None,
        "decoder_adjs_AVG": None,
        "encoder_adjs_SUM": np.zeros((past_steps, 51, 51)),
        "decoder_adjs_SUM": np.zeros((past_steps, 51, 51))
    }
    preds_dict = {ab: {} for ab in list_of_abbr}
    window_metric_1wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    window_metric_2wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    window_metric_3wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    window_metric_4wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    window_metric_5wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    window_metric_6wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": []
        }
        for ab in list_of_abbr
    }
    df_pred_unscaled = pd.DataFrame()
    for index, datat in enumerate(test_loader):
        print(os.path.basename(__file__), "index", index)
        (
            predictions__,
            attn_output_weights,
            encoder_sparse_weights,
            decoder_sparse_weights,
            encoder_adjs_output,
            decoder_adjs_output,
        ) = model_(datat, adjs, global_step=100000)
        start_date = datat["start_date"].detach().cpu().numpy().item()
        start_date_ = format_date(start_date)
        end_date = datat["end_date"].detach().cpu().numpy().item()
        end_date_ = format_date(end_date)
        csv_file_name = (
            ".." + f"/results/covid/{MODEL}_{INPUTID}_metric__{SEED}.csv")
        with open(csv_file_name, mode="a", encoding="utf-8") as file:
            for node_id, abbr in enumerate(list_of_abbr):
                print(os.path.basename(__file__), "node id, state", node_id,
                      abbr)

                model_.eval()
                predictions_ = predictions__[:, node_id, :, :]
                if abbr not in preds_dict:
                    preds_dict[abbr] = {}
                predictions_2 = predictions_.cpu().detach().numpy()
                predictions_to_scale = predictions_2.reshape(-1, 1)

                if scaler_name == "":
                    mean = standard_scaler_stats_dict[abbr]["mean"]
                    std = standard_scaler_stats_dict[abbr]["std"]
                    inverse_transformed_predictions = (predictions_to_scale *
                                                       std) + mean
                csv_path = ".." + f"/data/x_data_unscaled/{abbr}_label_unscaled.csv"
                if diff:
                    inverse_transformed_predictions = reconstruct_series_to_array(
                        inverse_transformed_predictions,
                        initial_values_date_dict[abbr], start_date_, end_date_,
                        "%Y-%m-%d", abbr)
                arr = inverse_transformed_predictions
                arr = np.where(arr <= 0., np.nan, arr)
                nans, x = nan_helper(arr)
                arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
                inverse_transformed_predictions = arr
                inverse_transformed_predictions = inverse_transformed_predictions
                combined_tar = get_values_in_date_range(
                    csv_path, start_date_, end_date_, "new_confirmed")
                actuals = combined_tar[past_steps:]
                combined_tar = np.array(combined_tar).reshape(-1, 1)
                if True:
                    dates_ = pd.date_range(start=start_date_,
                                           end=end_date_,
                                           freq="D")
                    if abbr not in preds_dict:
                        preds_dict[abbr] = {}
                    if start_date_ not in preds_dict[abbr]:
                        preds_dict[abbr][start_date_] = {}
                    preds_dict[abbr][start_date_][
                        "PRED"] = inverse_transformed_predictions.squeeze(
                        ).reshape(1, -1)
                    preds_dict[abbr][start_date_][
                        "ACTUAL_past"] = combined_tar[:past_steps, :].reshape(
                            1, -1)
                    preds_dict[abbr][start_date_][
                        "ACTUAL_future"] = combined_tar[
                            past_steps:, :].reshape(1, -1)
                    preds_dict[abbr][start_date_][
                        "DATES_past"] = dates_[:past_steps]
                    preds_dict[abbr][start_date_]["DATES_future"] = dates_[
                        past_steps:]
                    preds_dict[abbr][start_date_]["start_date"] = start_date_
                    preds_dict[abbr][start_date_]["end_date"] = end_date_
                    R = f"{save_path}/preds.json"
                    save_to_json(preds_dict, R)
                    from training_utils import window_evaluation
                    (RMSE_window_metric, MAE_window_metric, MAPE_window_metric,
                     CORR_window_metric) = window_evaluation(
                         config, actuals, inverse_transformed_predictions,
                         predictions_to_scale.squeeze().reshape(1, -1),
                         datat["outputs"][:,
                                          node_id, :, :].cpu().detach().numpy(
                                          ).squeeze().reshape(1, -1))
                    window_metric_1wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[0])
                    window_metric_1wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[0])
                    window_metric_1wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[0])
                    window_metric_1wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[0])
                    window_metric_2wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[1])
                    window_metric_2wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[1])
                    window_metric_2wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[1])
                    window_metric_2wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[1])
                    window_metric_3wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[2])
                    window_metric_3wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[2])
                    window_metric_3wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[2])
                    window_metric_3wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[2])
                    window_metric_4wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[3])
                    window_metric_4wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[3])
                    window_metric_4wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[3])
                    window_metric_4wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[3])
                    window_metric_5wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[4])
                    window_metric_5wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[4])
                    window_metric_5wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[4])
                    window_metric_5wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[4])
                    window_metric_6wk[abbr]["MAPE_LIST"].append(
                        MAPE_window_metric[5])
                    window_metric_6wk[abbr]["RMSE_LIST"].append(
                        RMSE_window_metric[5])
                    window_metric_6wk[abbr]["CORR_LIST"].append(
                        CORR_window_metric[5])
                    window_metric_6wk[abbr]["MAE_LIST"].append(
                        MAE_window_metric[5])
                    if node_id > 49 and index == 3:
                        metric_dicts_and_filenames = [
                            (window_metric_1wk, "window_metric_1wk.json"),
                            (window_metric_2wk, "window_metric_2wk.json"),
                            (window_metric_3wk, "window_metric_3wk.json"),
                            (window_metric_4wk, "window_metric_4wk.json"),
                            (window_metric_5wk, "window_metric_5wk.json"),
                            (window_metric_6wk, "window_metric_6wk.json")
                        ]
                        for metric_dict__, filename in metric_dicts_and_filenames:
                            for abbr__ in metric_dict__:
                                RMSE_mean = np.mean(
                                    metric_dict__[abbr__]["RMSE_LIST"])
                                CORR_mean = np.mean(
                                    metric_dict__[abbr__]["CORR_LIST"])
                                MAPE_mean = np.mean(
                                    metric_dict__[abbr__]["MAPE_LIST"])

                                MAE_mean = np.mean(
                                    metric_dict__[abbr__]["MAE_LIST"])
                                metric_dict__[abbr__]["RMSE"] = RMSE_mean
                                metric_dict__[abbr__]["CORR"] = CORR_mean
                                metric_dict__[abbr__]["MAPE"] = MAPE_mean
                                metric_dict__[abbr__]["MAE"] = MAE_mean
                                file_path = os.path.join(save_path, filename)
                                with open(file_path, "w", encoding="utf-8") as json_file:
                                    json.dump(serialize_dict(metric_dict__),
                                              json_file,
                                              indent=4)
                                if "6wk" in filename and not np.isnan(
                                        RMSE_mean):
                                    writer = csv.writer(file)
                                    name = MODEL
                                    writer.writerow([
                                        abbr__, name, RMSE_mean, MAE_mean,
                                        MAPE_mean, CORR_mean
                                    ])
                                    print(
                                        "State, Model, RMSE, MAE, MAPE, CORR",
                                        abbr__, name, RMSE_mean, MAE_mean,
                                        MAPE_mean, CORR_mean)
                                    file.flush()
    time.sleep(1)

    column_names = ["abbr", "name", "RMSE_mean", "MAE_mean", "MAPE_mean", "CORR_mean"]

    eval_df = pd.read_csv(csv_file_name, header=None, names=column_names, encoding="utf-8")

    averages = eval_df.iloc[:, 2:6]
    #print("averages", averages)
    averages = averages.mean(axis=0)#.values()
    #print("averages", averages)
    averages_df = pd.DataFrame([averages], columns=["RMSE_mean", "MAE_mean", "MAPE_mean", "CORR_mean"])
    print("averages_df", averages_df)
    averages_df.to_csv(csv_file_name.replace(".csv","_mean.csv"), index=False, encoding="utf-8")
    return config, preds_dict
