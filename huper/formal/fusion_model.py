#!/usr/bin/env python3
"""BrainX plus subject-specific learnable multi-kernel fusion parameters."""

from __future__ import annotations

import torch
from torch import nn


class MultiKernelBrainX(nn.Module):
    """Wrap BrainX and add per-subject target-fusion logits."""

    def __init__(
        self,
        *,
        brain,
        subjects: list[int],
        fusion_mode: str,
        num_kernels: int,
        num_tokens: int,
        feat_dim: int,
        kernels: list[int] | None = None,
        pixel_map_size: tuple[int, int] | None = None,
        init_std: float = 0.01,
        fusion_prior_alpha: float = 0.5,
    ):
        super().__init__()
        self.brain = brain
        self.subjects = [int(sub) for sub in subjects]
        self.fusion_mode = fusion_mode
        self.num_kernels = int(num_kernels)
        self.num_tokens = int(num_tokens)
        self.feat_dim = int(feat_dim)
        self.kernels = [int(kernel) for kernel in kernels] if kernels is not None else list(range(1, self.num_kernels + 1))
        self.pixel_map_size = tuple(pixel_map_size) if pixel_map_size is not None else None
        self.init_std = float(init_std)
        self.fusion_prior_alpha = float(fusion_prior_alpha)

        if len(self.kernels) != self.num_kernels:
            raise ValueError(f"Expected {self.num_kernels} kernels, got {len(self.kernels)}.")
        if any(kernel <= 0 for kernel in self.kernels):
            raise ValueError(f"All kernels must be positive, got {self.kernels}.")
        if self.init_std < 0:
            raise ValueError("init_std must be non-negative.")
        if self.fusion_prior_alpha < 0:
            raise ValueError("fusion_prior_alpha must be non-negative.")
        if self.fusion_mode != "token_wise":
            raise ValueError("The HuPer paper package supports token_wise fusion only.")
        if self.pixel_map_size is not None:
            raise ValueError("pixel_map_size must be None for token-wise HuPer.")

        self.fusion_logits = nn.ParameterDict()
        for sub in self.subjects:
            self.fusion_logits[f"fmri{sub}"] = nn.Parameter(self._new_logits())

    def _new_logits(self) -> torch.Tensor:
        shape = (self.num_kernels, self.num_tokens)

        logits = self._kernel_prior_logits(shape)
        if self.init_std > 0:
            logits = logits + torch.randn(shape) * self.init_std
        return logits

    def _kernel_prior_logits(self, shape: tuple[int, ...]) -> torch.Tensor:
        kernel_tensor = torch.tensor(self.kernels, dtype=torch.float32)
        prior = kernel_tensor.pow(-self.fusion_prior_alpha)
        prior = prior / prior.sum()
        view_shape = (self.num_kernels,) + (1,) * (len(shape) - 1)
        return torch.log(prior).view(view_shape).expand(shape).clone()

    def initial_kernel_prior_weights(self) -> dict[str, float]:
        kernel_tensor = torch.tensor(self.kernels, dtype=torch.float32)
        prior = kernel_tensor.pow(-self.fusion_prior_alpha)
        prior = prior / prior.sum()
        return {
            f"k{kernel}": float(weight)
            for kernel, weight in zip(self.kernels, prior.tolist())
        }

    def forward(self, x: torch.Tensor, modal: str = "fmri1") -> torch.Tensor:
        return self.brain(x, modal=modal)

    def fuse_image_features(self, kernel_features: torch.Tensor, modal: str) -> torch.Tensor:
        """Fuse cached [B,K,T,D] image features for a subject."""

        if kernel_features.ndim != 4:
            raise ValueError(f"kernel_features must be [B,K,T,D], got {tuple(kernel_features.shape)}.")
        if kernel_features.shape[1:] != (self.num_kernels, self.num_tokens, self.feat_dim):
            raise ValueError(
                "kernel_features shape mismatch: expected "
                f"[B,{self.num_kernels},{self.num_tokens},{self.feat_dim}], got {tuple(kernel_features.shape)}."
            )

        logits = self.fusion_logits[modal]
        weights = torch.softmax(logits, dim=0)
        weights = weights.to(device=kernel_features.device, dtype=kernel_features.dtype)
        return (kernel_features * weights[None, :, :, None]).sum(dim=1)

    def fusion_summary(self) -> dict:
        return {
            "fusion_mode": self.fusion_mode,
            "num_kernels": self.num_kernels,
            "kernels": self.kernels,
            "num_tokens": self.num_tokens,
            "feat_dim": self.feat_dim,
            "pixel_map_size": list(self.pixel_map_size) if self.pixel_map_size else None,
            "init_std": self.init_std,
            "fusion_prior_alpha": self.fusion_prior_alpha,
            "initial_kernel_prior_weights": self.initial_kernel_prior_weights(),
            "subjects": self.subjects,
            "logit_shapes": {key: list(value.shape) for key, value in self.fusion_logits.items()},
        }
