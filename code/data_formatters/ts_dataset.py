import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import os

device = "cuda"
torch.set_default_device(device)
torch.nn.Module.dump_patches = True


def timestamp_to_unix(timestamp):
    try:
        return timestamp.value // 10**9
    except:
        return pd.to_datetime(timestamp).value // 10**9


class TSDataset(Dataset):

    def __init__(
        self,
        CONTINUE,
        mode,
        id_col,
        static_cols,
        time_col,
        input_cols,
        target_col,
        total_steps,
        max_samples,
        input_size,
        past_steps,
        num_static,
        output_size,
        data,
        num_nodes,
        future_steps,
        prefix,
        INPUT_ID,
        data_name,
        scaler,
    ):
        self.CONTINUE = CONTINUE
        id_col = "categorical_id"
        self.mode = mode
        self.total_steps = total_steps
        self.input_size = input_size
        self.output_size = output_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.save_data = ".." + f"/data/x_data_pkl/{INPUT_ID}/{scaler}"
        if not os.path.exists(self.save_data):
            os.makedirs(self.save_data)
        self.file_paths = {
            "inputs": f"{self.save_data}/{self.mode}_inputs.pkl",
            "outputs": f"{self.save_data}/{self.mode}_outputs.pkl",
            "metadata": f"{self.save_data}/{self.mode}_metadata.pkl",
        }
        self.file_paths
        if (self.CONTINUE and not os.path.exists(self.file_paths["outputs"])
                or self.CONTINUE == False):
            data.sort_values(by=["date", 'categorical_id'], inplace=True)
            valid_sampling_locations = []
            split_data_map = {}
            self.identifiers = []
            self.tuple_dates = []
            for identifier, df in data.groupby(id_col):
                if identifier == "0.0":
                    continue
                self.identifiers.append(identifier)
                df = df.sort_values(by=["date"])
                df["new_confirmed"] = df["new_confirmed"].interpolate(
                    method='cubic').fillna(method="bfill").fillna(
                        method="ffill")
                df["new_confirmed_Y"] = df["new_confirmed_Y"].interpolate(
                    method='cubic').fillna(method="bfill").fillna(
                        method="ffill")
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
                dtype=torch.float,
                device="cuda")
            self.identifiers = list(set(self.identifiers))
            for i, start_idx in tqdm(enumerate(ranges),
                                     total=len(ranges),
                                     desc=f"■ Making {mode} data {INPUT_ID}"):
                for tt, identifier in enumerate(self.identifiers):
                    sliced = split_data_map[identifier].iloc[
                        start_idx - self.total_steps:start_idx]
                    S2 = sliced[input_cols[:-1]]
                    for col in S2.select_dtypes(include=["category"]).columns:
                        S2[col] = S2[col].cat.codes
                    for col in S2.select_dtypes(include=[np.number]).columns:
                        S2[col] = S2[col].astype(np.float32)
                    for col in S2.columns:
                        if S2[col].dtype == 'object':
                            S2[col] = S2[col].astype(str).astype(
                                'category').cat.codes
                    tensor = torch.tensor(
                        S2.to_numpy(),
                        device="cuda" if torch.cuda.is_available() else "cpu")
                    self.inputs_0[i, tt, :, :] = (
                        tensor.float().unsqueeze(0).unsqueeze(0))
                self.tuple_dates.append(
                    (sliced["date"].iloc[0], sliced["date"].iloc[-1]))
            self.inputs = self.inputs_0.clone()[:, :, :, 1:]
            self.outputs = self.inputs_0.clone()[:, :, self.past_steps:, 0:1]
            import pickle
            with open(self.file_paths["inputs"], "wb") as file:
                pickle.dump(self.inputs, file)
            with open(self.file_paths["outputs"], "wb") as file:
                pickle.dump(self.outputs, file)
            with open(self.file_paths["metadata"], "wb") as file:
                pickle.dump(self.tuple_dates, file)
        else:
            import pickle
            for key, file_path in self.file_paths.items():
                with open(file_path, "rb") as file:
                    data_loaded = pickle.load(file)
                    if key == "inputs":
                        self.inputs = data_loaded
                    elif key == "outputs":
                        self.outputs = data_loaded
                    elif key == "metadata":
                        self.tuple_dates = data_loaded

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        start_date = self.tuple_dates[index][0]
        end_date = self.tuple_dates[index][-1]
        start_date_unix = timestamp_to_unix(start_date)
        end_date_unix = timestamp_to_unix(end_date)
        sample = {
            "inputs": self.inputs[index],
            "outputs": self.outputs[index],
            "start_date": start_date_unix,
            "end_date": end_date_unix
        }
        return sample
