#!/usr/bin/env python3
"""Small training utilities used by HuPer."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_loss_fn(recon_loss: str):
    loss_functions = {
        "mse": F.mse_loss,
        "l1": F.l1_loss,
        "huber": F.smooth_l1_loss,
        "quantile": lambda x, y: torch.quantile(torch.abs(x - y), 0.9),
    }
    if recon_loss not in loss_functions:
        raise ValueError(f"Unrecognized loss type: {recon_loss}")
    return loss_functions[recon_loss]


def check_finite_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss).all():
        raise ValueError("Non-finite loss")


def seed_everything(seed: int = 0, cudnn_deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print("Note: not using cudnn.deterministic")


def print_parameter_count(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))


def adamw_parameter_groups(model, *, no_decay_names: list[str]):
    return [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(pattern in name for pattern in no_decay_names)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(pattern in name for pattern in no_decay_names)
            ],
            "weight_decay": 0.0,
        },
    ]


def plot_curve(values, label: str, output_path: str | Path, figsize=(10, 5)) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.plot(values, label=label)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
