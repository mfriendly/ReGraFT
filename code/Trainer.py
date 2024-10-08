import gc
import os

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

plt.rcParams.update({"font.size": 12})
import time

import numpy as np
import scienceplots
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from data_utils import *
from ReGraFT_model.regraft_model import ReGraFT as ReGraFT
from Tester import Test
from training_utils import *


import sys
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm


def initialize_optimizer(model, lr):
    return optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=0.01, lr=lr)


def initialize_model(config, adjs, device):
    if config["odd"]:
        model = ReGraFT(config, config["num_nodes"], adjs, device)
    else:
        model = ReGraFT(config, config["num_nodes"], adjs, device)
    model = model.to(torch.double)
    print("double")
    model = model.to(device)

    model.train()
    return model


def load_checkpoint(model, optimizer, config, device):
    try:
        checkpoint = torch.load(config["L_PATH"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(torch.double)
        print("double")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"■■■Loaded L_PATH{config['L_PATH']}")
    except Exception as e:
        print(f"Error loading checkpoint from {config['L_PATH']}: {e}")
        try:
            checkpoint = torch.load(config["C_PATH"], map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(torch.double)
            print("double")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = 0
            print(f"■■■Loaded C_PATH{config['C_PATH']}")
        except Exception as e:
            print(f"Error loading checkpoint from {config['C_PATH']}: {e}")
            start_epoch = 0
    return config, start_epoch, model, optimizer


def initialize_optimizer(model, lr):
    return optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=0.01, lr=lr)


def train_the_model(args, model_, train_loader, val_loader, adjs, config,
                    device, optimizer):

    best_val_loss = float("inf")
    global_step = 0
    MAX_STEPS = config["MAX_STEPS"] = args.MAX_STEPS
    V_MAX_STEPS = config["V_MAX_STEPS"] = args.V_MAX_STEPS

    MAX_STEPS = min(MAX_STEPS, len(train_loader))
    V_MAX_STEPS = min(V_MAX_STEPS, len(val_loader))
    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)

    accumulation_steps = 16
    epsilon = 1e-5
    scaler = GradScaler()

    epbar = tqdm(
        range(args.start_epoch, args.start_epoch + args.EPOCHS),
        total=args.EPOCHS,
        desc="train",
        leave=False,
        ncols=None,
    )

    for epoch in epbar:
        model_.train()
        epbar.set_description(
            f"Epoch {epoch}/{args.start_epoch + args.EPOCHS}")
        epoch_loss = []
        with torch.autograd.set_detect_anomaly(True):
            for tstep, tbatch in enumerate(
                    tqdm(train_loader,
                         total=MAX_STEPS,
                         desc="train",
                         leave=False,
                         ncols=None)):
                if tstep > config["MAX_STEPS"]:
                    break
                global_step += 1

                output, *_ = model_(tbatch, adjs, global_step)
                output = output.double().to(device)
                targ = tbatch["outputs"].double().to(device)

                if args.loss_type == "Huber":
                    loss = F.huber_loss(output,
                                        targ,
                                        reduction="mean",
                                        delta=args.huber_loss_delta)

                elif args.loss_type == "RMSE":
                    loss = torch.sqrt(F.mse_loss(output, targ))

                elif args.loss_type == "WeightedHuber":

                    short_term_loss = F.huber_loss(
                        output[:, :, :args.short_horizon, :],
                        targ[:, :, :args.short_horizon, :],
                        delta=args.huber_loss_delta,
                    )
                    long_term_loss = F.huber_loss(output,
                                                  targ,
                                                  delta=args.huber_loss_delta)

                    loss = args.huber_short_term_weight * short_term_loss + args.huber_long_term_weight * long_term_loss
                print("loss", loss)
                if args.corr_loss:
                    corr_loss = correlation_loss(output, targ)
                    loss = 0.5 * loss + 0.5 * corr_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN or Inf detected in loss. ")

                    sys.exit(1)

            loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)

            if (tstep + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss.append(loss.item() * accumulation_steps)

            if (tstep + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_epoch_loss = np.mean(epoch_loss)
        model_.eval()
        val_loss = []

        with torch.no_grad():
            for v, vbatch in enumerate(
                    tqdm(val_loader,
                         total=V_MAX_STEPS,
                         desc="valid",
                         leave=False,
                         ncols=None)):
                if v > config["V_MAX_STEPS"]:
                    break
                output, *_ = model_(vbatch, adjs, global_step=1000000)

                output, targ = output.double().to(
                    device), vbatch["outputs"].double().to(device)
                loss = torch.sqrt(F.mse_loss(output, targ) + epsilon)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        "NaN or Inf detected in validation loss. Skipping this batch."
                    )
                    continue

                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.C_PATH,
            )

        print(
            f"Epoch {epoch}, Train Loss: {avg_epoch_loss}, Val Loss: {avg_val_loss}"
        )
        gc.collect()
        torch.cuda.empty_cache()
        early_stopping(avg_val_loss, model_, f"{args.nation}_early_ckpt/")
        if early_stopping.early_stop:
            break

    return model_


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
    list_of_states, list_of_abbr, states2abbr, adjs = retrieve_metadata_of_nation(
        config, config["nation"], input_path, device)
    if make_stats:
        from mk_loader_stats import Prepare_Training_Inputs
    else:
        from mk_loader import Prepare_Training_Inputs
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
    ) = Prepare_Training_Inputs(config, args, list_of_states, list_of_abbr,
                                states2abbr, device)

    model_ = initialize_model(config, adjs, device)

    model2_ = None
    config = count_param_and_update_config(config, model_)
    save_path = config["save_path"]
    save_to_json(config, os.path.join(save_path, "config.json"))
    BATCHSIZE = config["BATCHSIZE"]

    train_loader, val_loader = create_loaders_from_dataset(
        config, train_dataset, val_dataset, BATCHSIZE, device)
    test_loader = prepare_test_loader(test_dataset, device)

    train_loader, val_loader = create_loaders_from_dataset(
        config, train_dataset, val_dataset, config["BATCHSIZE"], device)
    if args.EPOCHS != 0:

        config, args = retrieve_last_processed_give_lc_path(config, args)

        dir_path = os.path.dirname(config["C_PATH"])

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if True:

            start_time = time.time()
            optimizer = initialize_optimizer(model_, config["LR"])
            config, args.start_epoch, model_, optimizer = load_checkpoint(
                model_, optimizer, config, device)
            optimizer = initialize_optimizer(model_, config["LR"])
            model_ = train_the_model(args, model_, train_loader, val_loader,
                                     adjs, config, device, optimizer)
            end_time = time.time()
            training_time = end_time - start_time
            if "TrainingTime" in config:
                config["TrainingTime"].append(training_time)
            else:
                config["TrainingTime"] = [training_time]
    with torch.no_grad():
        if TEST:
            print("TESTING")

            config, preds_dict = Test(
                config,
                args,
                model_,
                test_loader,
                adjs,
                args.scaler_name,
                standard_scaler_stats_dict,
                args.diff,
                initial_values_date_dict,
                "6wk",
            )
            save_to_json(config,
                         os.path.join(config["save_path"], "config.json"))
            print("FINISHED TESTING")
