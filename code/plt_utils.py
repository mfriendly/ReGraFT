import json, os, glob

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import csv

import torch
import torch.nn.functional as F

OUT_PATH = "z_Variable_Importance"
os.makedirs(OUT_PATH, exist_ok=True)
INPUT_SIZE = 42
DIR_TO_PARSE = "results"

def init_csv(filename, labels, decoder):
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if decoder:
            csvwriter.writerow(["State"] + list(labels)[1:])
        else:
            csvwriter.writerow(["State"] + list(labels))

def init_csvs(filenames, labels):
    init_csv(filenames[0], labels, False)
    init_csv(filenames[1], labels, False)

def make_init_files(SOURCE, fnames):
    SOURCE_1 = f"{SOURCE}/_encoder_sparse/_matrix__02.npy"
    SOURCE_2 = f"{SOURCE}/_decoder_sparse/_matrix__02.npy"
    encoder_npy, decoder_npy = np.load(SOURCE_1), np.load(SOURCE_2)

    encoder_npy = encoder_npy[:, :, :].squeeze()
    decoder_npy = decoder_npy[:, :, :].squeeze()
    if decoder_npy.ndim == 2:
        decoder_npy = np.expand_dims(decoder_npy, axis=-1)
    ts, a, num_vars = encoder_npy.shape
    c_file = f"{SOURCE}/config.json"
    with open(c_file, "r") as f:
        config = json.load(f)

    variable_labels = config["input_cols"][1:-1]

    assert len(variable_labels) == num_vars

    init_csvs(fnames, variable_labels)
    return config, encoder_npy, decoder_npy, variable_labels

def write_to_csv(abbr, data, labels, filename):

    with open(filename, "a", newline="") as csvfile:

        csvwriter = csv.writer(csvfile)

        csvwriter.writerow([abbr] + list(data.numpy()))

def plot_bar(tensor, labels, filename, title):
    try:
        tensor = torch.tensor(tensor).squeeze()
    except:
        tensor = torch.from_numpy(tensor).squeeze()
    assert tensor.dim() == 1, "Tensor must be 1-dimensional"
    assert len(labels) == tensor.size(0), "Labels size must match tensor size"
    plt.figure(figsize=(8, 6))
    yyy = tensor.numpy().tolist()
    low = min(yyy)
    high = max(yyy)
    plt.ylim([low * 0.98, high * 1.02])
    plt.bar(labels, tensor.numpy(), color="skyblue")
    plt.xlabel("Variables")
    plt.ylabel("Values")
    plt.title(f"{title}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename + ".png")
    plt.close()

def masked_op(tensor, op="mean", dim=0, mask=None):
    if mask is None:
        mask = ~torch.isnan(tensor)
    masked = tensor.masked_fill(~mask, 0.0)
    summed = masked.sum(dim=dim)
    if op == "mean":
        return summed / mask.sum(dim=dim)
    elif op == "sum":
        return summed
    else:
        raise ValueError(f"unkown operation {op}")

def padded_stack(tensors, side="right", mode="constant", value=0):
    full_size = max([x.shape[-1] for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack([(F.pad(x, make_padding(full_size - x.shape[-1]), mode=mode, value=value)
          if full_size - x.shape[-1] > 0 else x) for x in tensors],
        dim=0,)
    return out

def interpret_output(out,
    reduction: str = "none",
    attention_prediction_horizon: int = 0,):
    batch_size = len(out["decoder_attention"])
    if isinstance(out["decoder_attention"], (list, tuple)):
        max_last_dimension = max(x.shape[-1] for x in out["decoder_attention"])
        first_elm = out["decoder_attention"][0]
        decoder_attention = torch.full((batch_size, *first_elm.shape[:-1], max_last_dimension),
            float("nan"),
            dtype=first_elm.dtype,
            device=first_elm.device,)
        for idx, x in enumerate(out["decoder_attention"]):
            decoder_length = out["decoder_lengths"][idx]
            decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
    else:
        decoder_attention = out["decoder_attention"].clone()
        decoder_mask = create_mask(out["decoder_attention"].shape[1],
                                   out["decoder_lengths"])
        decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")
    if isinstance(out["encoder_attention"], (list, tuple)):
        first_elm = out["encoder_attention"][0]
        encoder_attention = torch.full((batch_size, *first_elm.shape[:-1], INPUT_SIZE),
            float("nan"),
            dtype=first_elm.dtype,
            device=first_elm.device,)
        for idx, x in enumerate(out["encoder_attention"]):
            encoder_length = out["encoder_lengths"][idx]
            encoder_attention[idx, :, :, INPUT_SIZE -
                              encoder_length:] = x[..., :encoder_length]
    else:
        encoder_attention = out["encoder_attention"].clone()
        shifts = encoder_attention.shape[3] - out["encoder_lengths"]
        new_index = (torch.arange(encoder_attention.shape[3],
                                  device=encoder_attention.device)
                     [None, None, None].expand_as(encoder_attention) -
                     shifts[:, None, None, None]) % encoder_attention.shape[3]
        encoder_attention = torch.gather(encoder_attention,
                                         dim=3,
                                         index=new_index.long())
        if encoder_attention.shape[-1] < INPUT_SIZE:
            encoder_attention = torch.concat([torch.full((*encoder_attention.shape[:-1],
                            INPUT_SIZE - out["encoder_lengths"].max(),),
                        float("nan"),
                        dtype=encoder_attention.dtype,
                        device=encoder_attention.device,),
                    encoder_attention,],
                dim=-1,)
    attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
    attention[attention < 1e-5] = float("nan")
    encoder_length_histogram = integer_histogram(out["encoder_lengths"],
                                                 min=0,
                                                 max=INPUT_SIZE)
    decoder_length_histogram = integer_histogram(out["decoder_lengths"],
                                                 min=1,
                                                 max=INPUT_SIZE)
    encoder_variables = out["encoder_variables"].squeeze(-2).clone()
    encode_mask = create_mask(encoder_variables.shape[1],
                              out["encoder_lengths"])
    encoder_variables = encoder_variables.masked_fill(encode_mask.cpu().unsqueeze(-1), 0.0).sum(dim=1)
    out["encoder_lengths"] = torch.tensor(out["encoder_lengths"])
    encoder_variables /= (out["encoder_lengths"].where(out["encoder_lengths"] > 0,
        torch.ones_like(out["encoder_lengths"])).unsqueeze(-1))
    decoder_variables = out["decoder_variables"].squeeze(-2).clone()
    decode_mask = create_mask(decoder_variables.shape[1],
                              out["decoder_lengths"])
    decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
    out["decoder_lengths"] = torch.tensor(out["decoder_lengths"])
    decoder_variables /= out["decoder_lengths"].unsqueeze(-1)
    attention = masked_op(attention[:,
            attention_prediction_horizon,
            :,
            :INPUT_SIZE + attention_prediction_horizon,],
        op="mean",
        dim=1,)
    if reduction != "none":
        encoder_variables = encoder_variables.sum(dim=0)
        decoder_variables = decoder_variables.sum(dim=0)
        attention = masked_op(attention, dim=0, op=reduction)
    else:
        attention = attention / masked_op(attention, dim=1,
                                          op="sum").unsqueeze(-1)
    interpretation = dict(attention=attention.masked_fill(torch.isnan(attention), 0.0),
        encoder_variables=encoder_variables,
        decoder_variables=decoder_variables,
        encoder_length_histogram=encoder_length_histogram,
        decoder_length_histogram=decoder_length_histogram,)
    return interpretation

def create_mask(size: int, lengths: torch.LongTensor, inverse: bool = False):
    if inverse:
        return torch.tensor(torch.arange(torch.tensor(size))).unsqueeze(0) < torch.tensor(lengths).unsqueeze(-1)
    else:
        return torch.tensor(torch.arange(torch.tensor(size))).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(-1)

def integer_histogram(data, min, max):
    data = torch.tensor(data)
    uniques, counts = torch.unique(data, return_counts=True)
    if min is None:
        min = uniques.min()
    if max is None:
        max = uniques.max()
    hist = torch.zeros(max - min + 1, dtype=torch.long,
                       device=data.device).scatter(dim=0,
                                                   index=uniques - min,
                                                   src=counts)
    return hist

def Plot_Interpretation(config, encoder_npy, decoder_npy, list_of_abbr, SOURCE,
                        variable_labels):
    nation = config["nation"]
    device = "cpu"
    torch.set_default_device(device)
    fig_path = f"{SOURCE}/plt_attn"
    os.makedirs(fig_path, exist_ok=True)
    filename = "variable_importance.csv"
    fnames = [f"{fig_path}/encoder_{filename}", f"{fig_path}/decoder_{filename}"]

    init_csvs(fnames, variable_labels)
    for node_id, abbr in enumerate(list_of_abbr):
        E = encoder_npy[:, node_id, :]
        D = decoder_npy[:, node_id, :]

        out = {}

        out["decoder_variables"] = torch.tensor(D)[:, 1:].unsqueeze(1).unsqueeze(2).cpu()
        out["encoder_variables"] = torch.tensor(E).unsqueeze(1).unsqueeze(2).cpu()
        out["decoder_attention"] = torch.tensor(E)[:, 1:].unsqueeze(1).unsqueeze(2).cpu()
        out["encoder_attention"] = torch.tensor(D).unsqueeze(1).unsqueeze(2).cpu()
        out["decoder_lengths"] = np.array(range(1, INPUT_SIZE + 1))
        out["encoder_lengths"] = np.array(range(1, INPUT_SIZE + 1))
        interpretation = interpret_output(out, reduction="sum")
        process_and_save_interpretation(interpretation, abbr, variable_labels,
                                        fnames)
    df_encoder_varimp = pd.read_csv(fnames[0])
    mean_values = df_encoder_varimp.iloc[:, 1:].mean(axis=0)
    df4 = pd.DataFrame([mean_values], index=[nation])
    df4.to_csv(fnames[0].replace("oder_", "oder_avg_"))

    df_decoder_varimp = pd.read_csv(fnames[1])

    def shift_values_and_insert_nan(df):

        df.iloc[:, 1:] = df.iloc[:, 1:].shift(1, axis=1)

        df.iloc[:, 1] = np.nan

        return df

    df_decoder_varimp = shift_values_and_insert_nan(df_decoder_varimp)

    mean_values = df_decoder_varimp.iloc[:, 1:].mean(axis=0)

    df4 = pd.DataFrame([mean_values], index=[nation])
    df4.to_csv(fnames[1].replace("oder_", "oder_avg_"))

def Plot_VariableImportance_Globally(nation, current_timestamp):

    metric_dirs = glob.iglob(f"../{DIR_TO_PARSE}/covid/*_metric__*.csv")
    metric_dirs = [m for m in metric_dirs]
    tups = [m.split("/")[-1].replace("output/covid/", "").split("_metric__")[0]
        for m in metric_dirs]
    models = [m.rsplit("_", 1)[0] for m in tups]

    [int(m.split("_")[-1].split(".")[0]) for m in metric_dirs
        if " copy" not in m.split("_")[-1].split(".")[0]]

    with open("x_data_US/vars.json", "r") as file:
        all_cols = json.load(file)

    df_final = pd.DataFrame(columns=["Model"] + all_cols, index=models)
    fig, axs = plt.subplots(1, 2, figsize=(14, 15 * 0.25 + 2))
    for type in ["Encoder", "Decoder"]:
        if type == "Encoder":
            all_cols = ["new_confirmed"] + all_cols
        dfs = []
        files = glob.iglob(f"../{DIR_TO_PARSE}/covid/*/*/**/{type.lower()}_avg_variable_importance.csv",
            recursive=True,)
        print("files", files)
        for week_file in files:
            modeln = week_file.rsplit("/", 3)[-4]
            if os.path.exists(week_file):
                df_new = pd.DataFrame(columns=["Model"] + all_cols)

                df_sing = pd.read_csv(week_file).reset_index()

                try:
                    vals = list(df_sing.values[0][1:])
                    vals = [vals]

                    co = list(df_sing.columns)[1:]

                    co = [c.replace(f"_lag_{INPUT_SIZE}", "") for c in co]

                    df_new[co] = vals
                    df_new["Model"] = modeln

                    dfs.append(df_new[["Model"] + co])

                except:
                    pass

        with open("x_data_US/vars.json", "r") as file:
            cols = json.load(file)

        if type == "Encoder":
            cols_en = ["new_confirmed"] + cols

            df_out = pd.concat(dfs)
            df2 = df_out.sort_values(by=["Model"])

            cols_en = [c for c in cols_en if c in df2.columns]
            df2 = df2[cols_en]

            mean_values = df2.iloc[:, :].mean(axis=0)

            df3 = pd.DataFrame([mean_values], index=[nation])
            fig_path = f"{current_timestamp}/Variable Importance {type}"

            df3.to_csv(f"{OUT_PATH}/{fig_path}.csv".replace(" ", ""))

            variable_labels = df3.columns.tolist()

        else:
            df_out = pd.concat(dfs)

            df2 = df_out.sort_values(by=["Model"])
            cols = [c for c in cols if c in df2.columns]
            df2 = df2[cols]
            mean_values = df2.iloc[:, :].mean(axis=0)

            df3b = pd.DataFrame([mean_values], index=[nation], columns=cols)

            fig_path = f"{current_timestamp}/Variable Importance {type}"
            df3b.to_csv(f"{OUT_PATH}/{fig_path}.csv".replace(" ", ""))

def process_and_save_interpretation(interpretation, abbr, variable_labels,
                                    filenames):
    print("process_and_save_interpretation")

    write_to_csv(abbr, interpretation["encoder_variables"], variable_labels,
                 filenames[0])
    write_to_csv(abbr, interpretation["decoder_variables"],
                 variable_labels[1:], filenames[1])

def Prepare_Plot_VarImportance(nation, current_timestamp):

    filename = f"variable_importance.csv"
    fnames = [f"../data/x_data_aux/{nation}/encoder_{filename}",
        f"../data/x_data_aux/{nation}/decoder_{filename}",]
    id_df = pd.read_csv(f"../data/x_data_aux/{nation}/statemappings{nation}.csv")
    list_of_states = id_df["State"].tolist()
    list_of_abbr = id_df["Abbr"].tolist()
    dict(zip(list_of_states, list_of_abbr))

    if False:
        print("Sanity run")
        SOURCE = f"/home/minky/code_mk/ReGraFT_US_double/results/covid/US/ReGraFT-Top06-fusionFAP-H8-a64-11IF-WeightedHuber-s0.4-delta3.5-gcn2_42_42/11IF-Diff/999/"
        config, encoder_npy, decoder_npy, variable_labels = make_init_files(SOURCE, fnames)
        Plot_Interpretation(config, encoder_npy, decoder_npy, list_of_abbr,
                            SOURCE, variable_labels)

    all_items = glob.iglob(f"../{DIR_TO_PARSE}/covid/*/*/*/*//", recursive=True)

    directories_only = [item for item in all_items if os.path.isdir(item)]

    for SOURCE in directories_only:

        try:

            config, encoder_npy, decoder_npy, variable_labels = make_init_files(SOURCE, fnames)
            Plot_Interpretation(config, encoder_npy, decoder_npy, list_of_abbr,
                                SOURCE, variable_labels)
        except Exception as e:
            print(e)

    Plot_VariableImportance_Globally(nation, current_timestamp)

def plot_shared_x_two_y(args):
    values_1_path = f"../code/{OUT_PATH}/{args.nation}_var_importance_final_values.npy"
    labels_shared_path = f"../code/{OUT_PATH}/{args.nation}_var_importance_final_labels.json"
    csv_path2 = "../code/z_visualizations/plt_Hparam/heatmap_6.csv"

    values_1 = np.load(values_1_path)
    with open(labels_shared_path, "r") as f:
        labels = json.load(f)
    labels.reverse()
    labels_w_label = ["new_confirmed"] + labels

    values_1 = [0] + list(values_1)[::-1]

    df2 = pd.read_csv(csv_path2)
    values_df2 = list(df2["8"].values)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars = ax1.bar(labels_w_label,
                   values_1,
                   color="skyblue",
                   edgecolor="black",
                   linewidth=0.5)
    ax1.set_xlabel("Labels")
    ax1.set_ylabel("Importance", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax1.set_ylim(bottom=10)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 yval,
                 round(yval, 2),
                 ha="center",
                 va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(df2["Top_Variables"], values_df2, color="red")
    ax2.set_ylabel("Heatmap Value", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(bottom=min(values_df2) * 0.999)

    ax1.set_xticklabels(labels_w_label, rotation=90)

    plt.title("Bar and Line Plot with Shared X-Axis")
    fig.tight_layout()

    plt.savefig("shared_with_outline2.png")
    plt.close()
