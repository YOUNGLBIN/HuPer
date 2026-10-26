#!/usr/bin/env python3
"""Alignment losses used by HuPer."""

from __future__ import annotations

import torch


def token_alignment_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn) -> torch.Tensor:
    """Compute raw token alignment loss after a strict shape check."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}.")
    return loss_fn(pred.float(), target.float())
