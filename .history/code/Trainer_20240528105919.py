import pandas as pd
import types
import argparse
import json
from scipy import interpolate
import numpy as np
from torch.utils.data import DataLoader
import json, gc
from torch.utils.data import Subset
from datetime import timedelta

from mk_loader import make_loaders

from training_utils import *
from data_utils import *
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11})
import torch, os
import csv
import importlib, sys
import torch.optim as optim
from datetime import datetime
import glob
from torch.multiprocessing import set_start_method
import warnings

warnings.filterwarnings("ignore")
import string

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


def Train(model_, config, train_loader, val_loader, EPOCHS, MAX_STEPS,
          V_MAX_STEPS, L_PATH, adjs):
    try:
        try:
            checkpoint = torch.load(L_PATH)
        except Exception as e:
            print(e)
            pass
        model_.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.Adam(model_.parameters(),
                               betas=(0.9, 0.99),
                               eps=0.01,
                               lr=LR)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(e)
            pass
        start_epoch = checkpoint["epoch"] + 1
    except Exception as e:
        optimizer = optim.Adam(model_.parameters(),
                               betas=(0.9, 0.99),
                               eps=0.01,
                               lr=LR)
        start_epoch = 0
    best_val_loss = float("inf")
    losses = []
    val_losses = []
    global_step = 0
    model_.train()
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    config["PARAMS"] = params

    epbar = tqdm(
        range(start_epoch, start_epoch + EPOCHS),
        total=EPOCHS,
        desc="Training",
        leave=False,
        ncols=None,
    )
    early_stopping = EarlyStopping(patience=config["PATIENCE"], verbose=True)
    MAX_STEPS = min(MAX_STEPS, len(train_loader))
    for epoch in epbar:
        model_.train()
        epbar.set_description(f"Training Epoch {epoch}/{start_epoch + EPOCHS}")
        epoch_loss = []
        torch.cuda.empty_cache()
        with tqdm(
                enumerate(train_loader),
                total=MAX_STEPS,
                desc=f"Epoch - {epoch}",
                ncols=None,
        ) as pbar:
            for tstep, tbatch in pbar:
                if tstep > MAX_STEPS:
                    break
                global_step += 1
                (
                    output,
                    attn_output_weights,
                    encoder_sparse_weights,
                    decoder_sparse_weights,
                    encoder_adjs_output,
                    decoder_adjs_output,
                ) = model_(tbatch, adjs, global_step)
                targ = tbatch["outputs"].float()
                if True:  #config["LOSS"] == "RMSE":
                    loss = torch.sqrt(
                        torch.functional.F.mse_loss(output.float() / 1.0,
                                                    targ / 1.0))
                else:
                    corr_rmse_loss = CorrRMSELoss(alpha=0.1)

                    loss = corr_rmse_loss(output.float() / 1.0, targ / 1.0)
                if torch.isnan(loss):
                    sys.exit(1)
                optimizer.zero_grad()
                loss = loss.float()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                pbar.set_description(
                    f"Train loss: {np.mean(epoch_loss).item()}")
            torch.cuda.empty_cache()
            avg_epoch_loss = np.mean(epoch_loss)
            losses.append(avg_epoch_loss)
            model_.eval()
            val_loss = []
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad(), tqdm(
                    enumerate(val_loader),
                    total=min(len(val_loader), V_MAX_STEPS),
                    desc="Validation",
                    leave=False,
                    ncols=None,
            ) as vbar:
                for v, vbatch in vbar:
                    if v > V_MAX_STEPS:
                        break
                    (
                        output,
                        attn_output_weights,
                        encoder_sparse_weights,
                        decoder_sparse_weights,
                        encoder_adjs_output,
                        decoder_adjs_output,
                    ) = model_(vbatch, adjs, global_step=1000000)
                    aaa, bbb = (
                        output.float() / 1.0,
                        vbatch["outputs"].float() / 1.0,
                    )
                    import time
                    if True:  #con
                        loss = torch.sqrt(
                            torch.functional.F.mse_loss(
                                output.float() / 1.0,
                                vbatch["outputs"].float() / 1.0))
                    else:
                        corr_rmse_loss = CorrRMSELoss(alpha=0.1)

                        loss = corr_rmse_loss(output.float() / 1.0,
                                              vbatch["outputs"].float() / 1.0)
                    val_loss.append(loss.item())
                    pbar.set_postfix(loss=np.mean(val_loss).item())
                    pbar.set_description(
                        f"VAL loss: {np.mean(val_loss).item()}")
                gc.collect()
                torch.cuda.empty_cache()
                avg_val_loss = np.mean(val_loss)
                val_losses.append(avg_val_loss)
                CKPT_SAVE = config['CKPT_SAVE']
                C_PATH = config['C_PATH']
                PREFIX = config['PREFIX']
                if not os.path.exists(CKPT_SAVE):
                    os.makedirs(CKPT_SAVE, exist_ok=True)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    try:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model_.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()
                            },
                            C_PATH,
                            _use_new_zipfile_serialization=False,
                        )
                    except:
                        os.chmod(CKPT_SAVE, 0o700)
                        torch.save(
                            {
                                "model_state_dict": model_.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()
                            },
                            C_PATH,
                            _use_new_zipfile_serialization=False,
                        )
                gc.collect()
                torch.cuda.empty_cache()
                early_stopping(avg_val_loss, model_, "early_ckpt/")
                if early_stopping.early_stop:
                    break
        gc.collect()
        torch.cuda.empty_cache()
    try:
        try:
            checkpoint = torch.load(C_PATH)
            model_.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            print(e)
            pass
            checkpoint = torch.load(L_PATH)
            model_.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(e)
            pass
    except Exception as e:
        print(e)
        pass
    config["L_PATH"] = L_PATH
    config["C_PATH"] = C_PATH
    return model_, config


def interpolate_np(arr):
    indices = np.arange(len(arr))
    not_nan = ~np.isnan(arr)[0]
    linear_interpolator = interpolate.interp1d(indices[not_nan],
                                               arr[not_nan],
                                               kind='linear',
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


def Prepare_Training_Inputs(
        MODEL, SEED, diff, CONTINUE, MAKE_DFS, MAX_STEPS, V_MAX_STEPS,
        BATCHSIZE, EMBEDDING_DIM, past_steps, future_steps, total_steps,
        current_time, output_size, scaler_name, GRAPHS, graphs_str, GCN_DEPTH,
        num_static, target_col, unknown_real_x_target, holiday_cols,
        time_cat_cols_partial, time_idx_cols, time_cat_cols, pols_cols_Indices,
        pols_cols_C, pols_cols_E, pols_cols_H, pols_cols, lags,
        policy_lag_vars, gt_cols, gt_lags, gt_lag_vars, input_cols, INPUTID,
        DYN, CL, data_name, time_col, SPLIT, LOSSTYPE, stateUS, list_of_states,
        list_of_abbr, dict_of_states, start_attn, end_attn, ADAPTIVEGRAPH,
        patience, input_path, output_path):
    config = {}
    config['skip_connection'] = True
    config['adaptive_graph'] = ADAPTIVEGRAPH
    print("ADAPTIVEGRAPH", ADAPTIVEGRAPH)

    if diff:
        MODEL += 'Diff'

    config['end_attn'] = end_attn
    config['start_attn'] = start_attn
    config["scaler_name"] = scaler_name

    config["add_target_to_scale_vars"] = True

    if 'ReGraFT' in MODEL:
        config["save_attn_weight"] = True
        config["save_adj_weight"] = True
    else:
        config["save_attn_weight"] = False
        config["save_adj_weight"] = False
    config["save_transform_weight"] = False
    config["START"], config["END"] = "2020-03-01", "2022-03-01"
    config["dropout_type"] = "zoneout"
    config["fusion_mode"] = "none"
    config["version"] = ""
    config["gcn_depth"] = GCN_DEPTH
    config["gcn_alpha"] = 1.0
    config['diff'] = diff

    config['difference-Days'] = 1
    config['apply-difference-to-X'] = True
    config["unknown_real_x_target"] = unknown_real_x_target
    config["pol_lags"] = lags
    config["holiday_cols"] = holiday_cols
    config["policy_lag_vars"] = policy_lag_vars
    config["pols_cols"] = pols_cols
    config["gt_lag_vars"] = gt_lag_vars
    config["gt_cols"] = gt_cols
    config["gt_lags"] = gt_lags
    config["time_cat_cols"] = time_cat_cols
    config["input_cols"] = input_cols
    config["target_col"] = target_col
    config["EMBEDDING_DIM"] = EMBEDDING_DIM
    time_cat_cols = config["time_cat_cols"]
    config["LOSS"] = LOSSTYPE
    config["SEED"] = SEED
    config.update({
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "current_time": current_time,
        "BATCHSIZE": BATCHSIZE,
        "past_steps": past_steps,
        "future_steps": future_steps,
        "output_size": output_size,
        "dynamic": DYN,
        "SPLIT": SPLIT,
        "total_steps": total_steps,
        "CL": CL,
        "data_name": data_name,
        "MODEL": MODEL,
        "time_col": "date",
        "CONTINUE": CONTINUE,
        "PATIENCE": patience,
        "num_nodes": 51,
        "CL_STEP": 1000,
        "SPLIT": SPLIT,
        "INPUTID": INPUTID
    })

    if config["diff"]:
        INPUTID += "-Diff"
    config["INPUTID"] = INPUTID
    PREFIX = f"{MODEL}_{past_steps}_{future_steps}"
    PREFIX_fuzzy = f"*{MODEL}*{past_steps}_{future_steps}"
    config.update({
        "PREFIX": PREFIX,
        "PREFIX_fuzzy": PREFIX_fuzzy,
    })
    set_seed(SEED)
    save_path = os.path.join(output_path, data_name, PREFIX, INPUTID,
                             str(SEED))
    config["save_path"] = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        (train, valid, test, train_loader, val_loader, test_loader, config,
         dist_matrix, travel_matrix, input_cols, target_col, policy_lag_vars,
         holiday_cols, INPUTID, standard_scaler_stats_dict,
         standard_scaler_stats_dict_one, test_dataset,
         initial_values_date_dict,
         const_dict) = make_loaders(config,
                                    input_path,
                                    SEED,
                                    CONTINUE,
                                    data_name,
                                    diff=diff,
                                    aug=False,
                                    scaler_name=scaler_name)
    except Exception as e:
        print(e, "Train.py")
        import time
        time.sleep(5)
        CONTINUE = False
        (train, valid, test, train_loader, val_loader, test_loader, config,
         dist_matrix, travel_matrix, input_cols, target_col, policy_lag_vars,
         holiday_cols, INPUTID, standard_scaler_stats_dict,
         standard_scaler_stats_dict_one, test_dataset,
         initial_values_date_dict,
         const_dict) = make_loaders(config,
                                    input_path,
                                    SEED,
                                    CONTINUE,
                                    data_name,
                                    diff=diff,
                                    aug=False,
                                    scaler_name=scaler_name)
    standard_scaler_stats_dict
    save_path = config["save_path"]
    train_tmp, valid_tmp, test_tmp = train, valid, test
    input_size = config["input_size"]
    config["input_size"] = input_size
    config["attn_heads"] = 1
    if input_size % config["attn_heads"] != 0:
        config["attn_heads"] = 1
    best = {
        "lr": LR,
        "gcn_depth": 1,
        "GRAPHS": ["dist", "travel"],
        "dropout": 0.6,
        "gcn_dropout": 0.6
    }
    config.update(best)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    config["elu"] = True
    config["use_transform_skip"] = True
    config["version"] = ""
    GRAPHS = config["GRAPHS"]
    GRAPHS.reverse()
    graphs_str = "-".join(GRAPHS)

    config["BATCHSIZE"] = BATCHSIZE
    config["PREFIX"] = PREFIX
    current_time = config["current_time"]
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    config["PREFIX"] = PREFIX

    config["MODEL"] = MODEL
    save_path = config["save_path"]
    CKPT_SAVE = save_path + "/y_ckpt/"
    if not os.path.exists(CKPT_SAVE):
        os.makedirs(CKPT_SAVE)
    os.chmod(CKPT_SAVE, 0o700)
    PREFIX = config["PREFIX"]
    current_time = config["current_time"]

    last_processed_num = int(get_last_processed_number(CKPT_SAVE))
    print("➡ last_processed_num :", last_processed_num)
    last_processed_num = str(last_processed_num)
    C_PATH_num = int(last_processed_num) + 1
    save_last_processed_number(CKPT_SAVE, C_PATH_num)
    L_PATH = CKPT_SAVE + f"bests-v{last_processed_num}.pth"
    C_PATH = CKPT_SAVE + f"bests-v{C_PATH_num}.pth"
    config['CKPT_SAVE'] = CKPT_SAVE
    config["L_PATH"] = L_PATH
    config["C_PATH"] = C_PATH
    R = f"{save_path}/config.json"
    save_to_json(config, R)
    args = types.SimpleNamespace(**config)
    config["input_size"] = config["input_size"]
    return train, valid, test, train_loader, val_loader, test_loader, config, dist_matrix, travel_matrix, input_cols, target_col, policy_lag_vars, holiday_cols, INPUTID, standard_scaler_stats_dict, test_dataset, initial_values_date_dict


def train_regraft(TEST=True, input_path=None, output_path=None):

    print(
        "Available Adaptive Graph Generator versions: Fc, Attn, Pool, fusionFAP, fusionFA, fusionFP, fusionAP"
    )
    SEED = 15
    end_attn = True
    start_attn = True
    parser = argparse.ArgumentParser(description="COVID-19")
    parser.add_argument("--SEED", type=int, default=SEED)
    parser.add_argument('--start_attn', default=start_attn)
    parser.add_argument('--end_attn', default=end_attn)
    parser.add_argument('--ADAPTIVEGRAPH', type=str, default="Fc")
    args = parser.parse_args()
    ADAPTIVEGRAPH = args.ADAPTIVEGRAPH
    SEED = args.SEED
    end_attn = args.end_attn
    start_attn = args.start_attn
    GCN_DEPTH = 1
    MODEL = f'ReG{ADAPTIVEGRAPH}'
    EPOCHS = 1
    patience = 1
    diff = True
    CONTINUE = True
    MAKE_DFS = False
    if MAKE_DFS:
        CONTINUE = False
    MAX_STEPS = 300
    V_MAX_STEPS = 9
    BATCHSIZE = 4
    EMBEDDING_DIM = 1
    past_steps = 42
    future_steps = past_steps
    total_steps = past_steps + future_steps
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_size = 1
    scaler_name = ""
    GRAPHS = sorted(["travel", "dist"])
    GRAPHS.reverse()
    graphs_str = "-".join(GRAPHS)
    num_static = 0
    target_col = "new_confirmed_Y"
    unknown_real_x_target = ["new_confirmed"]
    holiday_cols = [
        "is_holiday", "is_holiday_lag_1", "is_holiday_lag_2",
        "is_holiday_lag_3"
    ]
    time_cat_cols_partial = [
        "categorical_week",
        "categorical_month",
        "categorical_day_of_week",
    ]
    time_idx_cols = [
        "week_from_start",
        'days_from_start',
    ]
    time_cat_cols = time_cat_cols_partial + time_idx_cols
    pols_cols_Indices = [
        "stringency_index", "government_response_index",
        "containment_health_index", "economic_support_index"
    ]
    pols_cols_C = [
        "school_closing", "workplace_closing", "cancel_public_events",
        "restrictions_on_gatherings", "public_transport_closing",
        "stay_at_home_requirements", "restrictions_on_internal_movement",
        "international_travel_controls"
    ]
    pols_cols_E = ["income_support", "debt_relief"]
    pols_cols_H = [
        "public_information_campaigns", "testing_policy", "contact_tracing",
        "facial_coverings", "vaccination_policy", "protection_elderly"
    ]
    pols_cols = [
        "restrictions_on_gatherings", "cancel_public_events",
        "international_travel_controls", "contact_tracing"
    ]
    lags = [past_steps]
    holiday_cols = sorted(holiday_cols)
    time_cat_cols = sorted(time_cat_cols)
    pols_cols = sorted(pols_cols)
    policy_lag_vars = []
    for pol in pols_cols:
        policy_lag_vars.append(pol + f"_lag_{past_steps}")
    gt_cols = []
    gt_lags = []
    gt_lag_vars = []
    for gt in gt_cols:
        gt_lag_vars.append(gt + "_lag_0")
    policy_lag_vars = sorted(policy_lag_vars)
    gt_lag_vars = sorted(gt_lag_vars)
    input_cols = ([target_col] + unknown_real_x_target + policy_lag_vars +
                  gt_lag_vars + holiday_cols + time_cat_cols +
                  ["categorical_id"])
    INPUTID = check_if_exp_id_exists_from_reading_json_then_create_one_if_not(
        input_cols, input_path)
    output_size = 1
    DYN = "dynamic"
    CL = True
    SEASON = ""
    data_name = "covid"
    time_col = "date"
    SPLIT = (0.7, 0.85)
    LOSSTYPE = "RMSE"
    stateUS = pd.read_csv(input_path + f"/x_data_aux/statemappingUS.csv")
    list_of_states = stateUS["State"].tolist()
    list_of_abbr = stateUS["Abbr"].tolist()
    dict_of_states = dict(zip(list_of_states, list_of_abbr))
    dist_matrix = np.load(input_path + f"/x_data_aux/matrix_0.npy")
    travel_matrix = np.load(input_path + f"/x_data_aux/matrix_1.npy")
    dist_matrix__ = torch.tensor(dist_matrix, dtype=torch.float64)
    travel_matrix__ = torch.tensor(travel_matrix, dtype=torch.float64)
    gdict = {"dist": dist_matrix__, "travel": travel_matrix__}
    adjs = []
    for gg in GRAPHS:
        adjs.append(gdict[gg].to(device))
    train, valid, test, train_loader, val_loader, test_loader, config, dist_matrix, travel_matrix, input_cols, target_col, policy_lag_vars, holiday_cols, INPUTID, standard_scaler_stats_dict, test_dataset, initial_values_date_dict = Prepare_Training_Inputs(
        MODEL, SEED, diff, CONTINUE, MAKE_DFS, MAX_STEPS, V_MAX_STEPS,
        BATCHSIZE, EMBEDDING_DIM, past_steps, future_steps, total_steps,
        current_time, output_size, scaler_name, GRAPHS, graphs_str, GCN_DEPTH,
        num_static, target_col, unknown_real_x_target, holiday_cols,
        time_cat_cols_partial, time_idx_cols, time_cat_cols, pols_cols_Indices,
        pols_cols_C, pols_cols_E, pols_cols_H, pols_cols, lags,
        policy_lag_vars, gt_cols, gt_lags, gt_lag_vars, input_cols, INPUTID,
        DYN, CL, data_name, time_col, SPLIT, LOSSTYPE, stateUS, list_of_states,
        list_of_abbr, dict_of_states, start_attn, end_attn, ADAPTIVEGRAPH,
        patience, input_path, output_path)
    from ReGraFT_model.regraft_model import ReGraFT as ReGraFT
    model_ = ReGraFT(config, 51, adjs)
    model_ = model_.to(device)
    model_.train()
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    config["PARAMS"] = params
    save_path = config['save_path']
    R = f"{save_path}/config.json"
    save_to_json(config, R)
    test_indices = np.array(
        [i for i in range(len(test_dataset)) if i % 7 == 2])
    test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        generator=torch.Generator(device="cuda"),
        drop_last=True,
    )
    if EPOCHS != 0:
        L_PATH = config['L_PATH']
        import time
        start_time = time.time()
        model_, config = Train(model_, config, train_loader, val_loader,
                               EPOCHS, MAX_STEPS, V_MAX_STEPS, L_PATH, adjs)
        end_time = time.time()
        training_time = end_time - start_time
        if 'TrainingTime' in config:
            config['TrainingTime'].append(training_time)
        else:
            config['TrainingTime'] = [training_time]
        R = f"{save_path}/config.json"
        save_to_json(config, R)
    with torch.no_grad():
        if TEST:
            print("TESTING")
            from Tester import Test
            config, preds_dict = Test(config, model_, test_loader, adjs,
                                      scaler_name, standard_scaler_stats_dict,
                                      diff, initial_values_date_dict)
            R = f"{save_path}/config.json"
            save_to_json(config, R)
            print("FINISHED TESTING")


if __name__ == "__main__":
    input_path = '../data'
    output_path = '../results'
    train_regraft(TEST=True, input_path=input_path, output_path=output_path)
