import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def get_tensor_from_dict(batch):
    input = batch["inputs"][:, :, :42, :]
    output = batch["inputs"][:, :, 42:, :0:1]

class ShapeStandardizer_4Donly:

    def __init__(self, expected: str, actual: str, batch: int, node: int,
                 time: int, feature: int):
        self.expected_shape_format = self.parse_shape_format(expected)
        self.actual_shape_format = self.parse_shape_format(actual)
        self.batch = batch
        self.node = node
        self.time = time
        self.feature = feature

    def parse_shape_format(self, shape_format: str):
        return tuple(shape_format.strip("() ").lower())

    def infer_shapes(self):
        current_shape = []
        expected_shape = []

        for dim in self.actual_shape_format:
            if dim == "b":
                current_shape.append(self.batch)
            elif dim == "n":
                current_shape.append(self.node)
            elif dim == "t":
                current_shape.append(self.time)
            elif dim == "f":
                current_shape.append(self.feature)

        for dim in self.expected_shape_format:
            if dim == "b":
                expected_shape.append(self.batch)
            elif dim == "n":
                expected_shape.append(self.node)
            elif dim == "t":
                expected_shape.append(self.time)
            elif dim == "f":
                expected_shape.append(self.feature)

        return tuple(current_shape), tuple(expected_shape)

    def check_tensor_shape(self, tensor: torch.Tensor, current_shape: tuple):
        if tensor.shape != current_shape:
            raise ValueError(
                f"Incorrect current shape: {tensor.shape}, expected: {current_shape}"
            )

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        current_shape, expected_shape = self.infer_shapes()
        tensor = self.detect_and_reshape(tensor, current_shape, expected_shape)
        self.check_tensor_shape(tensor, expected_shape)
        batch_size, nm_nodes, past_step, input_size = expected_shape
        reshaped_tensor = tensor.view(batch_size * nm_nodes, past_step,
                                      input_size)
        mean = reshaped_tensor.mean(dim=1, keepdim=True)
        std = reshaped_tensor.std(dim=1, keepdim=True)
        standardized_tensor = (reshaped_tensor - mean) / std
        return standardized_tensor.view(expected_shape)

    def inverse(self, standardized_tensor: torch.Tensor,
                original_tensor: torch.Tensor) -> torch.Tensor:
        current_shape, expected_shape = self.infer_shapes()
        standardized_tensor = self.detect_and_reshape(standardized_tensor,
                                                      current_shape,
                                                      expected_shape)
        self.check_tensor_shape(standardized_tensor, expected_shape)
        batch_size, nm_nodes, past_step, input_size = expected_shape
        reshaped_standardized = standardized_tensor.view(
            batch_size * nm_nodes, past_step, input_size)
        mean = original_tensor.view(batch_size * nm_nodes, past_step,
                                    input_size).mean(dim=1, keepdim=True)
        std = original_tensor.view(batch_size * nm_nodes, past_step,
                                   input_size).std(dim=1, keepdim=True)
        original = reshaped_standardized * std + mean
        return original.view(expected_shape)

    def detect_and_reshape(self, tensor: torch.Tensor, current_shape: tuple,
                           expected_shape: tuple) -> torch.Tensor:
        if len(tensor.shape) == 3:
            b, n, t = tensor.shape
            if self.actual_shape_format == ("b", "n", "t"):
                tensor = tensor.unsqueeze(-1)
            elif self.actual_shape_format == ("n", "t", "f"):
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                raise ValueError(
                    f"Unrecognized shape pattern for 3D input: {tensor.shape}")
        return tensor

class ShapeStandardizer:

    def __init__(self, expected: str, actual: str, batch: int, node: int,
                 time: int, feature: int):
        self.expected_shape_format = self.parse_shape_format(expected)
        self.actual_shape_format = self.parse_shape_format(actual)
        self.batch = batch
        self.node = node
        self.time = time
        self.feature = feature

    def parse_shape_format(self, shape_format: str):
        return tuple(shape_format.strip("() ").lower())

    def infer_shapes(self):
        current_shape = []
        expected_shape = []

        for dim in self.actual_shape_format:
            if dim == "b":
                current_shape.append(self.batch)
            elif dim == "n":
                current_shape.append(self.node)
            elif dim == "t":
                current_shape.append(self.time)
            elif dim == "f":
                current_shape.append(self.feature)

        for dim in self.expected_shape_format:
            if dim == "b":
                expected_shape.append(self.batch)
            elif dim == "n":
                expected_shape.append(self.node)
            elif dim == "t":
                expected_shape.append(self.time)
            elif dim == "f":
                expected_shape.append(self.feature)

        return tuple(current_shape), tuple(expected_shape)

    def check_tensor_shape(self, tensor: torch.Tensor, current_shape: tuple):
        if tensor.shape != current_shape:
            raise ValueError(
                f"Incorrect current shape: {tensor.shape}, expected: {current_shape}"
            )

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        current_shape, expected_shape = self.infer_shapes()

        if "f" in self.expected_shape_format and "f" in self.actual_shape_format:

            if self.expected_shape_format == ("b", "n", "f"):
                print(
                    "Warning: Target shape is BNF. Taking only the 0th dimension of the feature."
                )
                tensor = tensor[..., 0]

        tensor = self.detect_and_reshape(tensor, current_shape, expected_shape)
        self.check_tensor_shape(tensor, expected_shape)

        batch_size, nm_nodes, past_step, input_size = expected_shape
        reshaped_tensor = tensor.view(batch_size * nm_nodes, past_step,
                                      input_size)
        mean = reshaped_tensor.mean(dim=1, keepdim=True)
        std = reshaped_tensor.std(dim=1, keepdim=True)
        standardized_tensor = (reshaped_tensor - mean) / std
        return standardized_tensor.view(expected_shape)

    def inverse(self, standardized_tensor: torch.Tensor,
                original_tensor: torch.Tensor) -> torch.Tensor:
        current_shape, expected_shape = self.infer_shapes()
        standardized_tensor = self.detect_and_reshape(standardized_tensor,
                                                      current_shape,
                                                      expected_shape)
        self.check_tensor_shape(standardized_tensor, expected_shape)
        batch_size, nm_nodes, past_step, input_size = expected_shape
        reshaped_standardized = standardized_tensor.view(
            batch_size * nm_nodes, past_step, input_size)
        mean = original_tensor.view(batch_size * nm_nodes, past_step,
                                    input_size).mean(dim=1, keepdim=True)
        std = original_tensor.view(batch_size * nm_nodes, past_step,
                                   input_size).std(dim=1, keepdim=True)
        original = reshaped_standardized * std + mean
        return original.view(expected_shape)

    def detect_and_reshape(self, tensor: torch.Tensor, current_shape: tuple,
                           expected_shape: tuple) -> torch.Tensor:
        if len(tensor.shape) == 3:
            b, n, t = tensor.shape
            if self.actual_shape_format == ("b", "n", "t"):
                tensor = tensor.unsqueeze(-1)
            elif self.actual_shape_format == ("n", "t", "f"):
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                raise ValueError(
                    f"Unrecognized shape pattern for 3D input: {tensor.shape}")
        return tensor

def window_evaluation(config, label, outputs, label_unscaled, outputs_unscaled,
                      node_id):

    index = config["index"]

    label, outputs = label.reshape(1, -1), outputs.reshape(1, -1)
    windows = [7, 14, 21, 28, 35, 42]
    if config["finegrain_evaluation"]:
        future_steps = config["future_steps"]

        windows = [step for step in range(future_steps) if step % 3 == 0]

    if config["NRMSE_evaluation"]:
        metrics = [
            [NRMSE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [RMSE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [MAE(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [MAPE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [
                CORR_(label[:, i - 7:i], outputs[:, i - 7:i]) * 100
                for i in windows
            ],
        ]
    else:
        metrics = [
            [RMSE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [MAE(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [MAPE_(label[:, i - 7:i], outputs[:, i - 7:i]) for i in windows],
            [
                CORR_(label[:, i - 7:i], outputs[:, i - 7:i]) * 100
                for i in windows
            ],
        ]
    try:
        range_val = [[save_range_target(label[:, i - 7:i]) for i in windows]]
        os.makedirs(f"range_val/{index}", exist_ok=True)
        json_file_path = f"range_val/{index}/{str(node_id).zfill(2)}_range_val.json"
        range_val_serializable = [[
            val.tolist() if isinstance(val, np.ndarray) else val
            for val in sublist
        ] for sublist in range_val]

        with open(json_file_path, "w") as json_file:
            json.dump(range_val_serializable, json_file, indent=4)

    except Exception as e:
        print(f"Error saving range values: {str(e)}")
    return metrics

def prepare_test_loader(test_dataset, device):
    test_indices = np.array(
        [i for i in range(len(test_dataset)) if i % 7 == 2])
    test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
        drop_last=True,
    )
    return test_loader

def correlation_coefficient_loss(outputs, targets):

    mean_x = torch.mean(outputs, dim=2, keepdim=True)
    mean_y = torch.mean(targets, dim=2, keepdim=True)
    vx = outputs - mean_x
    vy = targets - mean_y
    cost = torch.sum(vx * vy, dim=2) / (torch.sqrt(torch.sum(vx**2, dim=2)) *
                                        torch.sqrt(torch.sum(vy**2, dim=2)))
    return torch.mean(cost)

class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})."
            )
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss

def RMSE_(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

def NRMSE_(predictions, targets):
    rmse = np.sqrt(((predictions - targets)**2).mean())
    range_val = targets.max() - targets.min()

    return rmse / range_val

def save_range_target(targets):

    range_val = targets.max() - targets.min()

    return range_val

def MAE(predictions, targets):
    r = np.abs(predictions - targets).mean()
    return r

def MAPE_(predictions, targets):
    return np.mean(np.abs((targets - predictions) / targets)) * 100

def CORR_(pred, true):
    true = np.atleast_1d(true)
    pred = np.atleast_1d(pred)
    true_mean = true.mean() if true.size > 0 else 0
    pred_mean = pred.mean() if pred.size > 0 else 0
    u = ((true - true_mean) * (pred - pred_mean)).sum()
    d = np.sqrt(((true - true_mean)**2).sum() * ((pred - pred_mean)**2).sum())
    if d == 0:
        return 0
    corr = u / d
    return corr if np.isscalar(corr) else corr.mean() if corr.size > 0 else 0

import torch

def correlation_loss(predictions, targets):
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)

    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)

    pred_std = torch.std(pred_flat)
    target_std = torch.std(target_flat)

    correlation = torch.mean(
        (pred_flat - pred_mean) *
        (target_flat - target_mean)) / (pred_std * target_std + 1e-8)

    corr_loss = 1 - correlation

    return corr_loss

def set_seed(seed=42, make_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
