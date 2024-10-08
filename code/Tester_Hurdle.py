import json

import numpy as np
import pandas as pd

LR = 4.245451262763638e-06
import matplotlib
import matplotlib.pyplot as plt

from data_utils import *

matplotlib.use("Agg")
import os
import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore")


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


current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
START, END = datetime(2020, 3, 1), datetime(2022, 3, 1)


def timestamp_to_unix(timestamp):
    return timestamp.value // 10**9


def Test(
    config,
    args,
    forecaster,
    MODEL_,
    model_,
    node_id,
    abbr,
    train,
    test_loader,
    scaler_name,
    standard_scaler_stats_dict,
    diff,
    initial_values_date_dict,
    window_metric_1wk,
    window_metric_2wk,
    window_metric_3wk,
    window_metric_4wk,
    window_metric_5wk,
    window_metric_6wk,
    preds_dict,
):
    print("Test")
    past_steps = future_steps = config["past_steps"]
    total_steps = past_steps * 2
    INPUTID = config["INPUTID"]
    SEED = config["SEED"]
    set_seed(SEED)
    nation = args.nation
    print("SEED", SEED)
    data_name = "covid"
    INPUTID = config["INPUTID"]
    data_name = config["data_name"]

    list_of_states, list_of_abbr, states2abbr, adjs = retrieve_metadata_of_nation(
        config, config["nation"], args.input_path, args.device)
    preds_US_dict = {"US": {}}

    with open(f"../data/x_data_aux/{nation}/Mapper_DaysFromStart2Date.json"
              ) as f:
        json.load(f)
    with open(f"../data/x_data_aux/{nation}/abbr2id.json") as ff:
        dict_ = json.load(ff)
    id_to_abbr = {v: k for k, v in dict_.items()}

    MODEL = MODEL_
    future_steps = past_steps
    config["MODEL"] = MODEL
    PREFIX = f"{MODEL}_{past_steps}_{future_steps}"
    f"*{MODEL}*{past_steps}_{future_steps}"
    save_path = os.path.join(f"../results", data_name, nation, PREFIX, INPUTID,
                             str(SEED))
    config["save_path"] = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_path = f"{save_path}/y_figs"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    CKPT_SAVE = save_path + "/y_ckpt/"
    if not os.path.exists(CKPT_SAVE):
        os.makedirs(CKPT_SAVE)
    os.chmod(CKPT_SAVE, 0o700)
    current_time = config["current_time"]
    CKPT_SAVE + f"{PREFIX}_best_{current_time}.pth"
    csv_file_name = f"../results/covid" + f"/{nation}/{MODEL}_{INPUTID}_metric__{SEED}.csv"
    for index, datat in enumerate(test_loader):

        config["index"] = index

        total_input = datat["inputs"].float()
        total_input_s = total_input.size()
        y_ = datat["outputs"].float()
        y = y_[:, node_id, :, :]
        x = total_input[:, node_id, :, :]

        device = "cpu"
        total_input = (torch.tensor(
            x, dtype=torch.float32).to(device)[:, :, 0].unsqueeze(-1).reshape(
                total_input_s[0], total_steps, -1))
        testBATCHSIZE = total_input_s[0]
        if "Naive" in MODEL:

            y = y_
            total_input_s = total_input.size()

            testBATCHSIZE = total_input_s[0]
            y = (torch.tensor(
                y, dtype=torch.float32).to(device)[:, node_id, :,
                                                   0].unsqueeze(-1).reshape(
                                                       testBATCHSIZE,
                                                       future_steps, -1))
            predictions = (total_input[:, past_steps, :].repeat(
                1, future_steps).reshape(testBATCHSIZE, future_steps, -1))

            abbr = list_of_abbr[node_id]
            predictions = predictions.cpu().detach().numpy()
            total_input = total_input.cpu().detach().numpy()
            predictions_ = predictions.reshape(1, future_steps, -1)
            print("predictions_", predictions_)
        elif "ARIMA" in MODEL.upper():
            x = torch.tensor(x,
                             dtype=torch.float32).to(device)[:, :past_steps,
                                                             0].unsqueeze(-1)
            y = torch.tensor(y,
                             dtype=torch.float32).to(device)[:, :,
                                                             0].unsqueeze(-1)
            total_input_s = total_input.shape
            total_input = (torch.tensor(
                total_input,
                dtype=torch.float32).to(device)[:, :, 0].unsqueeze(-1).reshape(
                    total_input_s[0], total_steps, -1))
            testBATCHSIZE = total_input_s[0]
            predictions_ = x
            predictions = torch.tensor(predictions_,
                                       dtype=torch.float32).to(device)
            predictions = predictions.reshape(testBATCHSIZE, future_steps, -1)
            predictions_ = predictions.cpu().detach().numpy()
        else:

            predictions = forecaster.predict(steps=future_steps)
            y = y.ravel()
            predictions__ = predictions.to_numpy()
            predictions_ = predictions__

        abbr = list_of_abbr[node_id]
        try:
            predictions__ = predictions_.to_numpy(dtype=float)
        except:
            predictions__ = predictions_
        mean = standard_scaler_stats_dict[abbr]["mean"]
        std = standard_scaler_stats_dict[abbr]["std"]
        for aaa in range(testBATCHSIZE):
            csv_path = f"../data/x_data_unscaled/{nation}/{str(node_id).zfill(2)}_{abbr}_label_unscaled.csv"
            start_date = datat["start_date"][aaa].detach().cpu().numpy().item()
            end_date = datat["end_date"][aaa].detach().cpu().numpy().item()
            start_date_ = format_date(start_date)
            end_date_ = format_date(end_date)
            combined_tar = get_values_in_date_range(csv_path, start_date_,
                                                    end_date_, "new_confirmed")
            actuals = combined_tar[past_steps:].reshape(-1, 1)
            combined_tar = np.array(combined_tar).reshape(-1, 1)
            predictions_to_scale = predictions__.reshape(-1, 42,
                                                         1)[aaa, :, :].reshape(
                                                             -1, 1)
            inverse_transformed_predictions = (predictions_to_scale *
                                               std) + mean
            inverse_transformed_predictions = inverse_transformed_predictions.reshape(
                -1, 1)
            if diff:
                inverse_transformed_predictions = reconstruct_series_to_array(
                    inverse_transformed_predictions,
                    initial_values_date_dict[abbr],
                    start_date_,
                    end_date_,
                    "%Y-%m-%d",
                    abbr,
                )

            arr = inverse_transformed_predictions
            arr = np.where(arr <= 0.0, np.nan, arr)
            nans, x = nan_helper(arr)
            arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
            inverse_transformed_predictions = arr
            inverse_transformed_predictions = inverse_transformed_predictions
            combined_tar = get_values_in_date_range(csv_path, start_date_,
                                                    end_date_, "new_confirmed")
            actuals = combined_tar[past_steps:]
            combined_tar = np.array(combined_tar).reshape(-1, 1)
            combined_tar_ = combined_tar.reshape(1, total_steps, -1)
            dates_ = pd.date_range(start=start_date_, end=end_date_, freq="D")
            if start_date_ not in preds_dict[abbr]:
                preds_dict[abbr][start_date_] = {}
            preds_dict[abbr][start_date_][
                "PRED"] = inverse_transformed_predictions[:, :].reshape(1, -1)
            preds_dict[abbr][start_date_][
                "ACTUAL_past"] = combined_tar_[:past_steps].reshape(1, -1)
            preds_dict[abbr][start_date_]["ACTUAL_future"] = combined_tar_[
                past_steps:].reshape(1, -1)
            preds_dict[abbr][start_date_]["DATES_past"] = dates_[:past_steps]
            preds_dict[abbr][start_date_]["DATES_future"] = dates_[past_steps:]
            preds_dict[abbr][start_date_]["start_date"] = start_date_
            preds_dict[abbr][start_date_]["end_date"] = end_date_
            R = f"{save_path}/preds.json"
            save_to_json(preds_dict, R)
            (NRMSE_window_metric, RMSE_window_metric, MAE_window_metric,
             MAPE_window_metric, CORR_window_metric) = window_evaluation(
                 config, actuals, inverse_transformed_predictions,
                 predictions_to_scale.squeeze().reshape(1,
                                                        -1), datat["outputs"]
                 [:, node_id, :, :].cpu().detach().numpy().squeeze().reshape(
                     1, -1), node_id)

            window_metric_1wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[0])

            window_metric_1wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[0])
            window_metric_1wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[0])
            window_metric_1wk[abbr]["CORR_LIST"].append(CORR_window_metric[0])
            window_metric_1wk[abbr]["MAE_LIST"].append(MAE_window_metric[0])

            window_metric_2wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[1])
            window_metric_2wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[1])
            window_metric_2wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[1])
            window_metric_2wk[abbr]["CORR_LIST"].append(CORR_window_metric[1])
            window_metric_2wk[abbr]["MAE_LIST"].append(MAE_window_metric[1])

            window_metric_3wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[2])
            window_metric_3wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[2])
            window_metric_3wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[2])
            window_metric_3wk[abbr]["CORR_LIST"].append(CORR_window_metric[2])
            window_metric_3wk[abbr]["MAE_LIST"].append(MAE_window_metric[2])

            window_metric_4wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[3])
            window_metric_4wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[3])
            window_metric_4wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[3])
            window_metric_4wk[abbr]["CORR_LIST"].append(CORR_window_metric[3])
            window_metric_4wk[abbr]["MAE_LIST"].append(MAE_window_metric[3])

            window_metric_5wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[4])
            window_metric_5wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[4])
            window_metric_5wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[4])
            window_metric_5wk[abbr]["CORR_LIST"].append(CORR_window_metric[4])
            window_metric_5wk[abbr]["MAE_LIST"].append(MAE_window_metric[4])

            window_metric_6wk[abbr]["MAPE_LIST"].append(MAPE_window_metric[5])
            window_metric_6wk[abbr]["NRMSE_LIST"].append(
                NRMSE_window_metric[5])
            window_metric_6wk[abbr]["RMSE_LIST"].append(RMSE_window_metric[5])
            window_metric_6wk[abbr]["CORR_LIST"].append(CORR_window_metric[5])
            window_metric_6wk[abbr]["MAE_LIST"].append(MAE_window_metric[5])

    return (config, preds_dict, window_metric_1wk, window_metric_2wk,
            window_metric_3wk, window_metric_4wk, window_metric_5wk,
            window_metric_6wk, csv_file_name, MODEL)
