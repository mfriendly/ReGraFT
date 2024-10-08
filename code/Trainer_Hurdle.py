import json
import threading
import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pmdarima as pm
import torch
import torch.nn as nn
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data_utils import *
from training_utils import *
matplotlib.use("Agg")
import csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

def interpolate_np(arr):
    import numpy as np
    from scipy import interpolate

    indices = np.arange(len(arr))
    print("indices", indices)
    not_nan = ~np.isnan(arr)[0]
    print("not_nan", not_nan)
    linear_interpolator = interpolate.interp1d(indices[not_nan],
                                               arr[not_nan],
                                               kind="linear",
                                               fill_value="extrapolate")
    arr_interpolated = linear_interpolator(indices)
    return arr_interpolated

def numpy_ffill(arr):
    arr = arr.squeeze()
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out

class ModelFactory:

    def __init__(self, seed):
        self.seed = seed

    def get_model(self, model_name):
        if model_name == "Naive":
            return None
        elif model_name == "Lasso":
            from sklearn.linear_model import Lasso

            return Lasso(random_state=self.seed)
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor

            return XGBRegressor(random_state=self.seed)
        elif model_name == "SVM":
            from sklearn.svm import SVR

            return SVR()
        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(random_state=self.seed)
        elif model_name == "LinearRegression":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        else:
            raise ValueError("Model not supported")

class Naive_thread(threading.Thread):

    def __init__(self, func, args=()):
        super(Naive_thread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.results = self.func(*self.args)

    def return_result(self):
        threading.Thread.join(self)
        return self.results

def _arima(seq, pred_len, bt, i):
    seq = seq.cpu().numpy()
    model = pm.auto_arima(seq)
    forecasts = model.predict(pred_len)
    return forecasts, bt, i

class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """

    def __init__(self, pred_len):
        super(Arima, self).__init__()
        self.pred_len = pred_len

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
        threads = []
        for bt, seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:, i]
                one_seq = Naive_thread(func=_arima,
                                       args=(seq, self.pred_len, bt, i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast, bt, i = every_thread.return_result()
            result[bt, :, i] = forcast
        return result

def run_training(config,
                 args,
                 TEST=True,
                 input_path=None,
                 output_path=None,
                 device=None,
                 make_stats=False):
    print(
        "Available Adaptive Graph Generator versions: Fc, Attn, Pool, fusionFAP, fusionFA, fusionFP, fusionAP"
    )

    from mk_loader import Prepare_Training_Inputs

    list_of_states, list_of_abbr, states2abbr, adjs = retrieve_metadata_of_nation(
        config, config["nation"], input_path, device)
    (
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
    ) = Prepare_Training_Inputs(config, args, list_of_states, list_of_abbr,
                                states2abbr, device)

    with torch.no_grad():
        test_indices = np.array(
            [i for i in range(len(test_dataset)) if i % 7 == 2])
        print("test_indices", len(test_indices))
        test_dataset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            generator=torch.Generator(device=args.device),
            drop_last=False,
        )

    SEED = config["SEED"]
    import types

    args = types.SimpleNamespace(**config)
    set_seed(SEED, make_deterministic=True)

    past_steps = config["past_steps"]
    future_steps = config["future_steps"]
    past_steps + future_steps
    preds_dict = {ab: {} for ab in list_of_abbr}
    window_metric_1wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_2wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_3wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_4wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_5wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_6wk = {
        ab: {
            "NRMSE": None,
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "NRMSE_LIST": [],
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    for node_id, abbr in enumerate(list_of_abbr):

        if abbr == "0.0":
            continue
        dfname = f"../data/x_data_df/{args.nation}/{str(node_id).zfill(2)}_{abbr}.csv"
        tqdm.write(f"reading {dfname}...".upper())

        if not os.path.exists(dfname):
            continue
        train_tmp, valid_tmp, test_tmp = train, valid, test
        train.reset_index(drop=True, inplace=True)
        train["id"] = train.index

        MODEL = args.MODEL
        from Tester_Hurdle import Test

        if "hurdle" in args.MODEL.lower():
            from hurdle_model import PoissonHurdleModel

            MODEL_, model_ = (args.MODEL, PoissonHurdleModel())
            forecaster = ForecasterAutoreg(regressor=model_, lags=past_steps)
            forecaster.fit(y=train["new_confirmed"])

        elif "arima" in args.MODEL.lower():
            from Arima_model import Arima

            model_ = Arima(pred_len=future_steps, order=(7, 1, 0))

            forecaster = None

            print("model_", model_)

        (
            config,
            preds_dict,
            window_metric_1wk,
            window_metric_2wk,
            window_metric_3wk,
            window_metric_4wk,
            window_metric_5wk,
            window_metric_6wk,
            csv_file_name,
            MODEL,
        ) = Test(
            config,
            args,
            forecaster,
            MODEL,
            model_,
            node_id,
            abbr,
            train,
            test_loader,
            args.scaler_name,
            standard_scaler_stats_dict,
            args.diff,
            initial_values_date_dict,
            window_metric_1wk,
            window_metric_2wk,
            window_metric_3wk,
            window_metric_4wk,
            window_metric_5wk,
            window_metric_6wk,
            preds_dict,
        )
        print(f"Calculating 6wek eval for {abbr} Finished")
    metric_dicts_and_filenames = [
        (window_metric_1wk, "window_metric_1wk.json"),
        (window_metric_2wk, "window_metric_2wk.json"),
        (window_metric_3wk, "window_metric_3wk.json"),
        (window_metric_4wk, "window_metric_4wk.json"),
        (window_metric_5wk, "window_metric_5wk.json"),
        (window_metric_6wk, "window_metric_6wk.json"),
    ]
    with open(csv_file_name, mode="w") as file:
        writer = csv.writer(file)
    for m_i, (metric_dict__,
              filename) in enumerate(metric_dicts_and_filenames):
        for j, abbr__ in enumerate(metric_dict__):

            print(f"Outer loop index: {m_i}, Inner loop index: {j}")

            NRMSE_list = [
                float(x) for x in metric_dict__[abbr__]["NRMSE_LIST"]
                if x is not None and not np.isnan(x)
            ]
            RMSE_list = [
                float(x) for x in metric_dict__[abbr__]["RMSE_LIST"]
                if x is not None and not np.isnan(x)
            ]
            corr_list = [
                float(x) for x in metric_dict__[abbr__]["CORR_LIST"]
                if x is not None and not np.isnan(x)
            ]
            MAPE_list = [
                float(x) for x in metric_dict__[abbr__]["MAPE_LIST"]
                if x is not None and not np.isnan(x)
            ]
            MAE_list = [
                float(x) for x in metric_dict__[abbr__]["MAE_LIST"]
                if x is not None and not np.isnan(x)
            ]

            NRMSE_mean = np.mean(NRMSE_list) if NRMSE_list else None
            RMSE_mean = np.mean(RMSE_list) if RMSE_list else None
            CORR_mean = np.mean(corr_list) if corr_list else None
            MAPE_mean = np.mean(MAPE_list) if MAPE_list else None
            MAE_mean = np.mean(MAE_list) if MAE_list else None

            metric_dict__[abbr__]["NRMSE"] = NRMSE_mean
            metric_dict__[abbr__]["RMSE"] = RMSE_mean
            metric_dict__[abbr__]["CORR"] = CORR_mean
            metric_dict__[abbr__]["MAPE"] = MAPE_mean
            metric_dict__[abbr__]["MAE"] = MAE_mean

            save_path = config["save_path"]
            file_path = os.path.join(save_path, filename)
            with open(file_path, "w") as json_file:
                json.dump(serialize_dict(metric_dict__), json_file, indent=4)

            csv_file_name = ".." + f"/results/covid/{args.nation}/{args.MODEL}_{args.INPUTID}_metric__{args.SEED}.csv"

            with open(csv_file_name, mode="a") as file:
                writer = csv.writer(file)
                if "6wk" in filename and not np.isnan(float(RMSE_mean)):
                    print("6wk in filename saving to ", csv_file_name)
                    name = MODEL

                    writer.writerow([
                        abbr__, name, NRMSE_mean, RMSE_mean, MAE_mean,
                        MAPE_mean, CORR_mean
                    ])

                    print("abbr, NRMSE, RMSE, MAE, MAPE, CORR".upper(), abbr__,
                          NRMSE_mean, RMSE_mean, MAE_mean, MAPE_mean,
                          CORR_mean)
                    R = f"{save_path}/preds.json"
                    with open(R, "w") as f:
                        json.dump(serialize_dict(preds_dict), f, indent=4)
                    R = f"{save_path}/config.json"
                    save_to_json(config, R)

    apply_mean_calculation(config, args, csv_file_name)
