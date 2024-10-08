import argparse
import os

import matplotlib.pyplot as plt
import scienceplots

from data_utils import *
from mk_loader import *
from training_utils import *
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import torch


def categorize_variables(variables):

    holiday_cols = []
    time_cat_cols_partial = []
    time_idx_cols = []
    pols_cols = []

    for var in variables:

        if var in [
                "is_holiday", "is_holiday_lag_1", "is_holiday_lag_2",
                "is_holiday_lag_3"
        ]:
            holiday_cols.append(var)
        elif var in [
                "categorical_week", "categorical_month",
                "categorical_day_of_week"
        ]:
            time_cat_cols_partial.append(var)
        elif var in ["week_from_start", "days_from_start"]:
            time_idx_cols.append(var)
        else:
            pols_cols.append(var)

    return sorted(holiday_cols), sorted(time_cat_cols_partial), sorted(
        time_idx_cols), sorted(pols_cols)


if __name__ == "__main__":

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7, 8"

    device = torch.device("cpu")

    input_path = "../data"
    output_path = "../results"

    print("device", device)

    input_path = "../data"
    output_path = "../results"

    print(
        "Available Adaptive Graph Generator versions: Fc, Attn, Pool, fusionFAP, fusionFA, fusionFP, fusionAP"
    )

    BATCHSIZE = 16
    nation = "US"
    SEED = 999
    start_attn = True
    end_attn = True
    adaptive_graph = "Fc"

    GCN_DEPTH = 1
    embedding_dim = 1
    past_steps = 42
    future_steps = 42
    total_steps = past_steps + future_steps
    patience = 1
    diff = True
    CONTINUE = 1
    MAKE_DFS = False

    EPOCHS = 0
    MAX_STEPS = 300
    V_MAX_STEPS = 9
    BATCHSIZE = 16
    L_PATH = ""
    C_PATH = ""
    use_wandb = False

    scaler_name = ""

    output_size = 1
    scaler_name = ""

    if MAKE_DFS:
        CONTINUE = False

    MODEL = f"Arima"

    MODEL = "Hurdle"
    if not diff:
        MODEL += "-XDiff"
    print("BATCHSIZE", BATCHSIZE)
    parser = argparse.ArgumentParser(description="COVID-19")
    parser.add_argument("--device", default=device)

    parser.add_argument("--MODEL", default=MODEL)
    parser.add_argument("--SEED", type=int, default=SEED)
    parser.add_argument("--start_attn", type=str2bool, default=start_attn)
    parser.add_argument("--end_attn", type=str2bool, default=end_attn)
    parser.add_argument("--adaptive_graph", type=str, default=adaptive_graph)

    parser.add_argument("--GCN_DEPTH", type=int, default=GCN_DEPTH)
    parser.add_argument("--embedding_dim", type=int, default=embedding_dim)
    parser.add_argument("--past_steps", type=int, default=past_steps)
    parser.add_argument("--future_steps", type=int, default=future_steps)
    parser.add_argument("--total_steps", type=int, default=total_steps)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--diff", type=str2bool, default=diff)
    parser.add_argument("--CONTINUE", type=str2bool, default=CONTINUE)
    parser.add_argument("--MAKE_DFS", type=str2bool, default=MAKE_DFS)

    parser.add_argument("--EPOCHS", type=int, default=EPOCHS)
    parser.add_argument("--MAX_STEPS", type=int, default=MAX_STEPS)
    parser.add_argument("--V_MAX_STEPS", type=int, default=V_MAX_STEPS)
    parser.add_argument("--BATCHSIZE", type=int, default=BATCHSIZE)

    parser.add_argument("--use_wandb", type=str2bool, default=use_wandb)

    parser.add_argument("--input_path", type=str, default="../data")
    parser.add_argument("--output_path", type=str, default="../results")
    parser.add_argument("--scaler_name", type=str, default="")
    parser.add_argument("--L_PATH",
                        type=str,
                        required=False,
                        help="Path to load checkpoint")
    parser.add_argument("--C_PATH",
                        type=str,
                        required=False,
                        help="Path to save checkpoint")
    parser.add_argument("--nation", type=str, default=nation)

    parser.add_argument("--range_val",
                        type=int,
                        default=1,
                        help="Number of top variables to use")
    parser.add_argument("--variables_json",
                        type=str,
                        default="vars.json",
                        help="JSON representation of")
    parser.add_argument("--rnn_type", type=str, default="")
    args = parser.parse_args()
    args.finegrain_evaluation = False
    args.NRMSE_evaluation = True

    device = str(args.device)
    device = torch.device(device)
    args.device = device

    with open(f"../data/x_data_aux/{nation}/config.json", "r") as f:
        config_init = json.load(f)
    args.num_nodes = config_init["num_nodes"]

    print(f"Running with SEED={args.SEED}, Device={args.device}, ")

    args.current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    args.input_path = input_path
    args.output_path = output_path

    args.LR = 4.245451262763638e-06
    args.num_static = 0
    args.target_col = "new_confirmed_Y"
    args.unknown_real_x_target = ["new_confirmed"]

    args.curriculum = True
    args.data_name = "covid"
    args.time_col = "date"
    args.SPLIT = (0.7, 0.85)
    args.LOSS = "RMSE"
    args.device = device

    args.dropout = 0.6
    args.adaptive_graph = "fusionFAP"

    args.make_stats = False

    args.JSON_PATH = "x_data_US"

    with open(join_paths(args.JSON_PATH, args.variables_json), "r") as f:
        important_variables = json.load(f)

    target_col = "new_confirmed_Y"
    unknown_real_x_target = ["new_confirmed"]
    holiday_cols = [
        "is_holiday", "is_holiday_lag_1", "is_holiday_lag_2",
        "is_holiday_lag_3"
    ]
    time_cat_cols_partial = [
        "categorical_week", "categorical_month", "categorical_day_of_week"
    ]
    args.time_idx_cols = ["days_from_start", "week_from_start"]

    selected_vars = important_variables[:args.range_val]

    args.holiday_cols, args.time_cat_cols_partial, _, args.pols_cols = categorize_variables(
        selected_vars)
    args.time_cat_cols = args.time_cat_cols_partial + args.time_idx_cols

    print("Unknown Real X Target:", args.unknown_real_x_target)
    print("Holiday Columns:", args.holiday_cols)
    print("Time Categorical Columns (Partial):", args.time_cat_cols_partial)
    print("Time Index Columns:", args.time_idx_cols)
    print("Policy Columns:", args.pols_cols)

    MODEL = args.MODEL

    with open(f"../data/x_data_aux/{nation}/config.json", "r") as f:
        config_init = json.load(f)
    args.num_nodes = config_init["num_nodes"]
    print("MODEL", MODEL)

    target_col = "new_confirmed_Y"
    unknown_real_x_target = ["new_confirmed"]
    holiday_cols = [
        "is_holiday", "is_holiday_lag_1", "is_holiday_lag_2",
        "is_holiday_lag_3"
    ]
    time_cat_cols_partial = [
        "categorical_week", "categorical_month", "categorical_day_of_week"
    ]
    args.time_idx_cols = ["days_from_start", "week_from_start"]

    selected_vars = important_variables[:args.range_val]

    args.holiday_cols, args.time_cat_cols_partial, _, args.pols_cols = categorize_variables(
        selected_vars)
    args.time_cat_cols = args.time_cat_cols_partial + args.time_idx_cols

    print("Unknown Real X Target:", args.unknown_real_x_target)
    print("Holiday Columns:", args.holiday_cols)
    print("Time Categorical Columns (Partial):", args.time_cat_cols_partial)
    print("Time Index Columns:", args.time_idx_cols)
    print("Policy Columns:", args.pols_cols)

    args.MODEL = f"Hurdle-Top{str(args.range_val).zfill(2)}"
    print(f"Running {args.MODEL} with Device={device}")

    args.policy_lag_vars = [pol + f"_lag_42" for pol in args.pols_cols]

    args.input_cols = ([args.target_col] + args.unknown_real_x_target +
                       args.policy_lag_vars + args.holiday_cols +
                       args.time_cat_cols + ["categorical_id"])
    print("Input columns:", args.input_cols)

    args.INPUTID = check_if_exp_id_exists_from_reading_json_then_create_one_if_not(
        args.input_cols, args.input_path, args.nation)
    print("args.INPUTID", args.INPUTID)
    args.output_size = 1

    args.curriculum = True
    args.data_name = "covid"
    args.time_col = "date"
    args.SPLIT = (0.7, 0.85)
    args.LOSSTYPE = "RMSE"
    args.device = device

    config = vars(args)

    from Trainer_Hurdle import run_training

    args.EPOCHS = 20
    run_training(config,
                 args,
                 TEST=True,
                 input_path=args.input_path,
                 output_path=args.output_path,
                 device=device)
