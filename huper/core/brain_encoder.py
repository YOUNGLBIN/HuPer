#!/usr/bin/env python3
# Adapted from UMBRAE model.py (author notice: Weihao Xia), Apache-2.0.
# Modified for HuPer and the anonymous review supplement; see THIRD_PARTY_NOTICES.md.
"""Brain encoders used by HuPer."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from huper.core.perceiver_resampler import PerceiverResampler


class Perceiver(nn.Module):
    def __init__(self, patch_embed_dim: int = 1024, hidden_size: int = 1024, num_latents: int = 256):
        super().__init__()
        self.ln_vision = nn.LayerNorm(patch_embed_dim)
        self.llm_proj = nn.Linear(patch_embed_dim, hidden_size)
        self.perceiver = PerceiverResampler(
            dim=patch_embed_dim,
            dim_head=96,
            depth=6,
            heads=16,
            num_latents=num_latents,
            num_media_embeds=1,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.ln_vision(image_features)
        inputs_llm = self.perceiver(image_features)
        return self.llm_proj(inputs_llm)


class BrainX(nn.Module):
    """Cross-subject fMRI encoder."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        num_latents: int = 256,
        use_token: bool = False,
        use_norm: bool = False,
        act_first: bool = False,
    ):
        super().__init__()
        self.subs = [1, 2, 5, 7]
        self.num_voxels = {
            1: 15724,
            2: 14278,
            3: 15226,
            4: 13153,
            5: 13039,
            6: 17907,
            7: 12682,
            8: 14386,
        }
        self.use_token = use_token

        norm_func = partial(nn.LayerNorm, normalized_shape=num_latents)
        act_fn = nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.lin1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        self.token = nn.ParameterDict()
        for sub in self.subs:
            modal = f"fmri{sub}"
            if not use_norm:
                self.lin1[modal] = nn.Linear(1, num_latents)
            else:
                self.lin1[modal] = nn.Sequential(
                    nn.Linear(1, num_latents),
                    *[item() for item in act_and_norm],
                    nn.Dropout(0.5),
                )
            self.lin2[modal] = nn.Linear(self.num_voxels[sub], hidden_dim)

            if self.use_token:
                self.token[modal] = nn.Parameter(torch.empty([1, 5, hidden_dim]))
                nn.init.normal_(self.token[modal], std=0.02)

        self.perceiver = Perceiver(
            patch_embed_dim=hidden_dim,
            hidden_size=out_dim,
            num_latents=num_latents,
        )

    def forward(self, x: torch.Tensor, modal: str = "fmri1") -> torch.Tensor:
        x = x.unsqueeze(1).transpose(1, 2)
        x = self.lin1[modal](x)
        x = x.transpose(1, 2)
        x = self.lin2[modal](x)
        if self.use_token:
            token = self.token[modal].repeat(x.size(0), 1, 1)
            x = torch.cat([x, token], dim=1)
        return self.perceiver(x)


class BrainXS(nn.Module):
    """Single-subject fMRI encoder."""

    def __init__(self, in_dim: int = 15724, hidden_dim: int = 1024, out_dim: int = 1024, num_latents: int = 256):
        super().__init__()
        self.lin1 = nn.Linear(1, num_latents)
        self.lin2 = nn.Linear(in_dim, hidden_dim)
        self.perceiver = Perceiver(
            patch_embed_dim=hidden_dim,
            hidden_size=out_dim,
            num_latents=num_latents,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).transpose(1, 2)
        x = self.lin1(x)
        x = x.transpose(1, 2)
        x = self.lin2(x)
        return self.perceiver(x)
