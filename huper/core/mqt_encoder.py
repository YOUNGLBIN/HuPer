#!/usr/bin/env python3
"""Frozen MQT-LLaVA image encoder without image augmentation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch import nn

from huper.core.paths import add_mqt_repo_to_path, resolve_hf_model_path


def image_processor_size_to_hw(size_value, default: int = 336) -> Tuple[int, int]:
    if size_value is None:
        return default, default
    if isinstance(size_value, int):
        return size_value, size_value
    if isinstance(size_value, (tuple, list)) and len(size_value) == 2:
        return int(size_value[0]), int(size_value[1])

    height = getattr(size_value, "height", None)
    width = getattr(size_value, "width", None)
    if height is None and hasattr(size_value, "get"):
        height = size_value.get("height", None)
    if width is None and hasattr(size_value, "get"):
        width = size_value.get("width", None)
    if height is not None and width is not None:
        return int(height), int(width)

    shortest_edge = getattr(size_value, "shortest_edge", None)
    if shortest_edge is None and hasattr(size_value, "get"):
        shortest_edge = size_value.get("shortest_edge", None)
    if shortest_edge is not None:
        shortest_edge = int(shortest_edge)
        return shortest_edge, shortest_edge

    size_value = int(size_value)
    return size_value, size_value


def infer_tgt_size(num_patches: int) -> Tuple[int, int]:
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square target size from {num_patches} vision tokens.")
    return side, side


def resize_images(images: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        images,
        size=image_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def patch_transformers_for_mqt() -> None:
    import transformers.modeling_utils as hf_modeling_utils
    import transformers.pytorch_utils as hf_pytorch_utils

    if not hasattr(hf_modeling_utils, "apply_chunking_to_forward"):
        hf_modeling_utils.apply_chunking_to_forward = hf_pytorch_utils.apply_chunking_to_forward
    if not hasattr(hf_modeling_utils, "prune_linear_layer"):
        hf_modeling_utils.prune_linear_layer = hf_pytorch_utils.prune_linear_layer
    if not hasattr(hf_modeling_utils, "find_pruneable_heads_and_indices"):

        def _unused_find_pruneable_heads_and_indices(*args, **kwargs):
            raise NotImplementedError("find_pruneable_heads_and_indices is not used here.")

        hf_modeling_utils.find_pruneable_heads_and_indices = _unused_find_pruneable_heads_and_indices


@dataclass(frozen=True)
class MQTCrossAttentionOutput:
    """MQT pre-projection tokens and their query-to-patch attention.

    ``cross_attention`` keeps the head dimension and has shape
    ``[batch, heads, queries, vision_patches]``.
    """

    pre_proj_tokens: torch.Tensor
    cross_attention: torch.Tensor
    target_grid_size: Tuple[int, int]


@torch.no_grad()
def query_abstractor_pre_proj(
    query_abstractor,
    image_features,
    num_visual_tokens,
    tgt_size,
    *,
    return_attention: bool = False,
):
    from llava.model.multimodal_projector.builder import get_abs_pos, get_matry_n

    abstractor_dtype = query_abstractor.ln_v.weight.dtype
    image_features = image_features.to(dtype=abstractor_dtype)
    pos_embed = get_abs_pos(query_abstractor.pos_embed, tgt_size).to(
        device=image_features.device,
        dtype=abstractor_dtype,
    )

    x = image_features.permute(1, 0, 2)
    batch_size = x.shape[1]

    matry_n = get_matry_n(num_visual_tokens)
    query = query_abstractor.query[:matry_n].to(dtype=abstractor_dtype)
    query = query_abstractor._repeat(query, batch_size)
    query_pos = query_abstractor.pos_embed[:matry_n].unsqueeze(1).to(
        device=image_features.device,
        dtype=abstractor_dtype,
    )

    output, attention = query_abstractor.attn(
        query_abstractor.ln_q(query + query_pos),
        query_abstractor.ln_k(query_abstractor._repeat(pos_embed, batch_size)),
        query_abstractor.ln_v(x),
        attn_mask=None,
        need_weights=True,
        average_attn_weights=not return_attention,
    )
    tokens = output.permute(1, 0, 2)
    if return_attention:
        return tokens, attention
    return tokens


class MQTPreProjEncoder(nn.Module):
    """MQT-LLaVA vision tower plus query-abstractor pre-projection tokens."""

    feature_space = "mqt"

    def __init__(
        self,
        model_path: str,
        repo_path: str,
        *,
        vision_model_path: str | None = None,
        num_tokens: int = 256,
        model_dtype: str = "float16",
        tgt_size: Optional[Tuple[int, int]] = None,
        local_files_only: bool = True,
    ):
        super().__init__()
        add_mqt_repo_to_path(repo_path)
        patch_transformers_for_mqt()

        from llava.model.multimodal_encoder.builder import build_vision_tower
        from llava.model.multimodal_projector.builder import build_matryoshka_query_transformer

        self.model_path = resolve_hf_model_path(model_path)
        self.num_tokens = int(num_tokens)
        if not 1 <= self.num_tokens <= 256:
            raise ValueError("mqt_num_tokens must be in [1, 256].")
        self.tgt_size = tuple(tgt_size) if tgt_size is not None else None
        self.dtype = getattr(torch, model_dtype)

        config_path = self.model_path / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            self.mqt_config = SimpleNamespace(**json.load(f))
        self.mqt_config.local_files_only = local_files_only
        if vision_model_path:
            self.mqt_config.mm_vision_tower = str(resolve_hf_model_path(vision_model_path))

        self.vision_tower = build_vision_tower(self.mqt_config, delay_load=True)
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model()
        self.vision_tower.eval()

        self.query_abstractor = build_matryoshka_query_transformer(self.mqt_config)
        query_state = torch.load(self.model_path / "query_abstractor.bin", map_location="cpu")
        query_state = {
            key.split("query_abstractor.", 1)[1]: value
            for key, value in query_state.items()
            if "query_abstractor." in key
        }
        missing, unexpected = self.query_abstractor.load_state_dict(query_state, strict=False)
        if missing or unexpected:
            print(f"Loaded query_abstractor with missing={missing}, unexpected={unexpected}")
        self.query_abstractor.eval()

        self.image_processor = self.vision_tower.image_processor
        crop_size = getattr(self.image_processor, "crop_size", None)
        if crop_size is None:
            crop_size = getattr(self.image_processor, "size", 336)
        self.image_size = image_processor_size_to_hw(crop_size, default=336)
        self.image_aspect_ratio = getattr(self.mqt_config, "image_aspect_ratio", "square")

        self.register_buffer(
            "image_mean",
            torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)

    @property
    def target_dim(self) -> int:
        return int(self.query_abstractor.proj.shape[0])

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        target_device = kwargs.get("device")
        if target_device is None:
            for arg in args:
                if isinstance(arg, (str, int, torch.device)):
                    target_device = arg
                    break
        if target_device is not None:
            self.vision_tower.to(device=target_device, dtype=self.dtype)
            self.query_abstractor.to(device=target_device, dtype=self.dtype)
        return result

    def pad_to_square(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_aspect_ratio != "pad":
            return images
        _, _, height, width = images.shape
        if height == width:
            return images

        size = max(height, width)
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        padded = mean.expand(images.shape[0], 3, size, size).clone()
        top = (size - height) // 2
        left = (size - width) // 2
        padded[:, :, top : top + height, left : left + width] = images
        return padded

    def preprocess_tensor_images(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            raise ValueError("HuPer MQTPreProjEncoder does not support training-time image augmentation.")

        images = images.float()
        if images.max() > 2:
            images = images / 255.0

        images = self.pad_to_square(images)
        images = resize_images(images, self.image_size)

        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def _preprocess_for_vision(self, images: torch.Tensor, training: bool) -> torch.Tensor:
        pixel_values = self.preprocess_tensor_images(images, training=training)
        return pixel_values.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        pixel_values = self._preprocess_for_vision(images, training=training)

        vision_features = self.vision_tower(pixel_values)
        abstractor_device = next(self.query_abstractor.parameters()).device
        vision_features = vision_features.to(abstractor_device)
        tgt_size = self.tgt_size if self.tgt_size is not None else infer_tgt_size(vision_features.shape[1])
        tokens = query_abstractor_pre_proj(
            self.query_abstractor,
            vision_features,
            num_visual_tokens="256",
            tgt_size=tgt_size,
        )
        return tokens[:, : self.num_tokens].detach().to(device=images.device)

    @torch.no_grad()
    def forward_with_cross_attention(
        self,
        images: torch.Tensor,
        training: bool = False,
    ) -> MQTCrossAttentionOutput:
        """Encode images and expose the MQT query-to-CLIP-patch attention.

        The existing :meth:`forward` API remains unchanged. This method is
        intended for analysis code that needs the direct query abstractor
        cross-attention without CLIP self-attention rollout.
        """

        pixel_values = self._preprocess_for_vision(images, training=training)
        vision_features = self.vision_tower(pixel_values)
        abstractor_device = next(self.query_abstractor.parameters()).device
        vision_features = vision_features.to(abstractor_device)
        tgt_size = self.tgt_size if self.tgt_size is not None else infer_tgt_size(vision_features.shape[1])
        tokens, cross_attention = query_abstractor_pre_proj(
            self.query_abstractor,
            vision_features,
            num_visual_tokens="256",
            tgt_size=tgt_size,
            return_attention=True,
        )

        return MQTCrossAttentionOutput(
            pre_proj_tokens=tokens[:, : self.num_tokens].detach().to(device=images.device),
            cross_attention=cross_attention[:, :, : self.num_tokens].detach(),
            target_grid_size=tuple(tgt_size),
        )
