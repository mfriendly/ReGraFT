import argparse
import os

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

plt.rcParams.update({"font.size": 12})
import os
import warnings
from datetime import datetime

import scienceplots

from data_utils import *
from mk_loader import *
from training_utils import *

warnings.filterwarnings("ignore")
import os
import time

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
    print("Available devices ", torch.cuda.device_count())
    device = torch.device("cuda:0")

    input_path = "../data"
    output_path = "../results"

    print(
        "Available Adaptive Graph Generator versions: Fc, Attn, Pool, fusionFAP, fusionFA, fusionFP, fusionAP"
    )

    BATCHSIZE = 16
    nation = "US"
    SEED = 31
    start_attn = True
    end_attn = True

    gcn_depth = 2
    embedding_dim = 1
    past_steps = 42
    future_steps = 42
    total_steps = past_steps + future_steps
    patience = 1
    diff = True
    CONTINUE = 1
    MAKE_DFS = 0

    EPOCHS = 0
    MAX_STEPS = 300
    V_MAX_STEPS = 9
    L_PATH = ""
    C_PATH = ""
    scaler_name = ""

    output_size = 1
    scaler_name = ""
    GRAPHS = sorted(["dist", "travel"])
    GRAPHS.reverse()
    graphs_str = "-".join(GRAPHS)
    if MAKE_DFS:
        CONTINUE = False

    MODEL = f"Ours"
    print("BATCHSIZE", BATCHSIZE)

    parser = argparse.ArgumentParser(description="COVID-19")
    parser.add_argument("--device", default=device)

    parser.add_argument("--GRAPHS", default=GRAPHS)
    parser.add_argument("--MODEL", default=MODEL)
    parser.add_argument("--SEED", type=int, default=SEED)
    parser.add_argument("--start_attn", type=str2bool, default=start_attn)
    parser.add_argument("--end_attn", type=str2bool, default=end_attn)

    parser.add_argument("--curriculum", type=str2bool, default=True)

    parser.add_argument("--gcn_depth", type=int, default=gcn_depth)
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

    parser.add_argument("--TEST", action="store_true")
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
    parser.add_argument("--attn_heads", type=int, default=8)

    parser.add_argument("--range_val",
                        type=int,
                        default=6,
                        help="Number of top variables to use")
    parser.add_argument("--variables_json",
                        type=str,
                        default="vars.json",
                        help="JSON representation of")
    parser.add_argument(
        "--adaptive_graph",
        type=str,
        default="fusionFAP",
        choices=[
            "fusionFAP", "fusionFA", "fusionAP", "fusionFP", "Fc", "Attn",
            "Pool"
        ],
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="Huber",
        choices=["RMSE", "Huber", "MultiRMSE", "WeightedHuber"])
    args = parser.parse_args()
    args.finegrain_evaluation = False
    args.NRMSE_evaluation = True

    args.loss_type = "WeightedHuber"
    args.corr_loss = False
    if args.loss_type == "Huber":
        args.huber_loss = True
        args.huber_loss_delta = 10.0

    elif args.loss_type == "WeightedHuber":
        args.huber_short_term_weight = 0.4
        args.huber_long_term_weight = 1.0 - args.huber_short_term_weight
        args.short_horizon = 14
        args.huber_loss_delta = 3.5
    elif args.loss_type == "MultiRMSE":
        args.short_horizon = 7
        args.short_horizon_ratio = (8, 2)
        short_horizon_ratio_string_list = [
            str(i) for i in args.short_horizon_ratio
        ]
        args.short_horizon_ratio_string = "".join(
            short_horizon_ratio_string_list)

    device = str(args.device)
    device = torch.device(device)
    args.device = device

    with open(f"../data/x_data_aux/{nation}/config.json", "r") as f:
        config_init = json.load(f)
    args.num_nodes = config_init["num_nodes"]
    torch.cuda.set_device(device)
    args.num_static_graph = len(GRAPHS)

    print("Current cuda device ", torch.cuda.current_device())

    print(f"Running with SEED={args.SEED}, Device={args.device}, ")

    args.current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    args.hidden_size = 4

    args.attn_hidden_dim = 32
    args.num_attention_layers = 2
    args.adaptive_graph_hidden = 64
    args.mha_hidden = 32
    args.transform_attn = False
    print("args.attn_heads", args.attn_heads)
    args.input_path = input_path
    args.output_path = output_path
    args.curriculum_step = 100

    args.temperature = 0.1
    args.LR = 4.245451262763638e-06
    args.num_static = 0
    args.target_col = "new_confirmed_Y"
    args.unknown_real_x_target = ["new_confirmed"]

    args.curriculum = True
    args.data_name = "covid"
    args.time_col = "date"
    args.SPLIT = (0.7, 0.85)

    args.device = device

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

    args.MODEL = f"ReGraFT-Top{str(args.range_val).zfill(2)}"
    print(f"Running {args.MODEL} with Device={device}")

    args.MODEL += "-" + args.adaptive_graph
    args.MODEL += "-H" + str(args.attn_heads)
    args.MODEL += "-a" + str(args.adaptive_graph_hidden)

    args.policy_lag_vars = [pol + f"_lag_42" for pol in args.pols_cols]

    args.input_cols = ([args.target_col] + args.unknown_real_x_target +
                       args.policy_lag_vars + args.holiday_cols +
                       args.time_cat_cols + ["categorical_id"])
    print("Input columns:", args.input_cols)

    args.INPUTID = check_if_exp_id_exists_from_reading_json_then_create_one_if_not(
        args.input_cols, args.input_path, args.nation)
    args.MODEL += "-" + args.INPUTID

    if args.loss_type == "Huber":
        args.MODEL += "-Huber"
        args.MODEL += "" + str(float(args.huber_loss_delta)).replace(".", "")
    elif args.loss_type == "WeightedHuber":
        args.MODEL += "-WeightedHuber"
        args.MODEL += "-s" + str(args.huber_short_term_weight)
        args.MODEL += "-delta" + str(args.huber_loss_delta)
    elif args.loss_type == "RMSE":
        args.MODEL += "-RMSE"
    if args.corr_loss:
        args.MODEL += "-CorrLoss"
    args = update_args_odd_input_size(args)
    print("args ODD", args.odd)

    act = {
        "act_gcn": "relu",
        "act_rnn": "selu",
        "act_encoder": "selu",
        "act_decoder": "selu",
    }
    args.output_size = 1

    op = {
        "LR": 4.245451262763638e-06,
        "dropout": 0.6,
        "gcn_depth": 2,
        "adaptive_graph_hidden": 64
    }

    args.gcn_depth = 2
    if args.diff:

        args.INPUTID += "-Diff"

    if args.scaler_name != "":

        args.INPUTID += args.scaler_name

    args.MODEL += "-gcn" + str(args.gcn_depth)

    from Trainer import run_training

    print("EPOCHS", args.EPOCHS)
    args.stage = ", "
    config = vars(args)
    config.update(op)
    config.update(act)

    run_training(config,
                 args,
                 TEST=True,
                 input_path=args.input_path,
                 output_path=args.output_path,
                 device=device)
