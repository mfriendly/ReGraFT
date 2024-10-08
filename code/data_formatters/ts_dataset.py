import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def save_data(file, filename):
    torch.save(file, filename)


torch.nn.Module.dump_patches = False


def timestamp_to_unix(timestamp):
    try:
        return timestamp.value // 10**9
    except:
        return pd.to_datetime(timestamp).value // 10**9


def mapper(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


class TSDataset(Dataset):

    def __init__(self, config, mode, data):

        id_col = "categorical_id"

        input_cols = config["input_cols"]

        total_steps = config["total_steps"]
        max_samples = config["max_samples"]
        input_size = config["input_size"]
        past_steps = config["past_steps"]

        output_size = config["output_size"]
        num_nodes = config["num_nodes"]
        future_steps = config["future_steps"]

        INPUT_ID = config["INPUTID"]
        config["data_name"]
        config["scaler_name"]
        nation = config["nation"]
        self.device = config["device"]
        self.CONTINUE = config["CONTINUE"]
        self.mode = mode
        self.total_steps = total_steps
        self.input_size = input_size
        self.output_size = output_size
        self.past_steps = past_steps
        self.future_steps = future_steps

        self.save_data = f"../data/x_data_pkl/{nation}/{INPUT_ID}/"

        if not os.path.exists(self.save_data):
            os.makedirs(self.save_data)
        self.file_paths = {
            "inputs": f"{self.save_data}/{self.mode}_inputs.pt",
            "outputs": f"{self.save_data}/{self.mode}_outputs.pt",
            "metadata": f"{self.save_data}/{self.mode}_metadata.pt",
        }
        aux_path = config["input_path"] + f"/x_data_aux/{nation}/"

        if self.CONTINUE and not os.path.exists(
                self.file_paths["outputs"]) or self.CONTINUE == False:
            data.sort_values(by=["date", "categorical_id"], inplace=True)
            valid_sampling_locations = []
            split_data_map = {}
            self.identifiers = []
            self.tuple_dates = []

            for identifier, df in data.groupby(by=[id_col]):
                if identifier == "0.0":
                    continue
                self.identifiers.append(identifier)
                df = df.sort_values(by=["date"])
                df["new_confirmed"] = (df["new_confirmed"].interpolate(
                    method="cubic").fillna(method="bfill").fillna(
                        method="ffill"))
                df["new_confirmed_Y"] = (df["new_confirmed_Y"].interpolate(
                    method="cubic").fillna(method="bfill").fillna(
                        method="ffill"))
                df = df[["date"] + input_cols]
                num_entries = df.shape[0]
                valid_sampling_locations = [
                    int(self.total_steps + i)
                    for i in range(num_entries - self.total_steps + 1)
                ]
                split_data_map[identifier] = df
                valid_sampling_locations = list(set(valid_sampling_locations))
                ranges = valid_sampling_locations
                max_samples = len(valid_sampling_locations)
            self.inputs_0 = torch.zeros(
                (max_samples, num_nodes, self.total_steps,
                 self.input_size + 1),
                dtype=torch.double,
                device=self.device)
            self.identifiers = list(set(self.identifiers))

            dict_emb = {}
            for i, start_idx in tqdm(enumerate(ranges),
                                     total=len(ranges),
                                     desc=f"■ Making {mode} data {INPUT_ID}"):
                for tt, identifier in enumerate(self.identifiers):
                    sliced = split_data_map[identifier].iloc[
                        start_idx - self.total_steps:start_idx]
                    S2 = sliced[input_cols[:-1]]

                    for col in S2.columns:
                        if S2[col].dtype == "bool":
                            S2[col] = S2[col].astype(int)

                        elif S2[col].dtype.name == "category" or S2[
                                col].dtype == "object":
                            S2[col] = S2[col].astype("category").cat.codes

                    for col in S2.select_dtypes(include=["category"]).columns:
                        S2[col] = S2[col].cat.codes
                    for col in S2.select_dtypes(include=[np.number]).columns:
                        S2[col] = S2[col].astype(np.float32)

                    for col in S2.columns:
                        if S2[col].dtype == "object":
                            S2[col] = S2[col].astype(str).astype(
                                "category").cat.codes
                    S2np = S2.to_numpy()

                    tensor = torch.tensor(S2np,
                                          dtype=torch.double,
                                          device=self.device if
                                          torch.cuda.is_available() else "cpu")
                    self.inputs_0[i, tt, :, :] = tensor.double().unsqueeze(
                        0).unsqueeze(0)

                self.tuple_dates.append(
                    (sliced["date"].iloc[0], sliced["date"].iloc[-1]))

            self.inputs = self.inputs_0.clone()[:, :, :, 1:]
            self.outputs = self.inputs_0.clone()[:, :, self.past_steps:, 0:1]
            torch.save(self.inputs, self.file_paths["inputs"])
            torch.save(self.outputs, self.file_paths["outputs"])
            torch.save(self.tuple_dates, self.file_paths["metadata"])
        else:

            self.inputs = torch.load(self.file_paths["inputs"],
                                     map_location="cpu")
            self.outputs = torch.load(self.file_paths["outputs"],
                                      map_location="cpu")
            self.tuple_dates = torch.load(self.file_paths["metadata"],
                                          map_location="cpu")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        start_date = self.tuple_dates[index][0]
        end_date = self.tuple_dates[index][-1]
        start_date_unix = timestamp_to_unix(start_date)
        end_date_unix = timestamp_to_unix(end_date)
        sample = {
            "inputs": self.inputs[index].double().to(self.device),
            "outputs": self.outputs[index].double().to(self.device),
            "start_date": start_date_unix,
            "end_date": end_date_unix,
        }
        return sample
