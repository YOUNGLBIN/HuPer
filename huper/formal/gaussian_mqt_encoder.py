#!/usr/bin/env python3
"""Frozen MQT encoder wrapper for multi-kernel Gaussian blur features."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from huper.core.paths import DEFAULT_MQT_MODEL_PATH, DEFAULT_MQT_REPO_PATH, setup_import_paths
from huper.formal.blur import validate_gaussian_kernels


class MultiKernelGaussianMQTEncoder:
    """Encode every image under every Gaussian blur kernel."""

    def __init__(
        self,
        *,
        kernels: list[int],
        device: str,
        mqt_repo_path: str | Path = DEFAULT_MQT_REPO_PATH,
        mqt_model_path: str = DEFAULT_MQT_MODEL_PATH,
        vision_model_path: str | None = None,
        mqt_num_tokens: int = 256,
        mqt_model_dtype: str = "float16",
        mqt_tgt_size: list[int] | None = None,
        local_files_only: bool = True,
    ):
        setup_import_paths(mqt_repo_path)
        from huper.core.mqt_encoder import MQTPreProjEncoder

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU for MQT image features.")
            device = "cpu"

        self.device = torch.device(device)
        self.kernels = list(kernels)
        validate_gaussian_kernels(self.kernels)
        self.encoder = MQTPreProjEncoder(
            model_path=mqt_model_path,
            repo_path=str(mqt_repo_path),
            vision_model_path=vision_model_path,
            num_tokens=mqt_num_tokens,
            model_dtype=mqt_model_dtype,
            tgt_size=tuple(mqt_tgt_size) if mqt_tgt_size else None,
            local_files_only=local_files_only,
        ).to(self.device)
        self.encoder.eval()
        self._gaussian_kernel_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
        self.freeze()

    @property
    def token_dim(self) -> int:
        return int(self.encoder.target_dim)

    @property
    def num_tokens(self) -> int:
        return int(self.encoder.num_tokens)

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.encoder.image_size)

    def freeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def prepare_pixels(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True).float()
        if images.max() > 2:
            images = images / 255.0
        images = self.encoder.pad_to_square(images)

        from huper.core.mqt_encoder import resize_images

        return resize_images(images, self.encoder.image_size)

    def gaussian_blur_stack(self, prepared_pixels: torch.Tensor) -> torch.Tensor:
        """Return Gaussian-blurred images shaped [kernel,batch,channel,height,width].

        This is a batched torch implementation of OpenCV GaussianBlur with
        ``sigma=0``. It keeps the operation on GPU when ``prepared_pixels`` is
        on GPU, avoiding the previous PIL/cv2 per-image loop.
        """

        if prepared_pixels.ndim != 4:
            raise ValueError(f"prepared_pixels must be [B,C,H,W], got {tuple(prepared_pixels.shape)}.")

        prepared_pixels = prepared_pixels.to(self.device, non_blocking=True)
        input_dtype = prepared_pixels.dtype
        compute_dtype = input_dtype if prepared_pixels.device.type == "cuda" else torch.float32
        pixels = prepared_pixels.to(dtype=compute_dtype).clamp(0, 1)

        blurred_by_kernel = [self._batch_gaussian_blur(pixels, kernel) for kernel in self.kernels]
        return torch.stack(blurred_by_kernel, dim=0).to(dtype=input_dtype)

    def _batch_gaussian_blur(self, pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size == 1:
            return pixels

        pad = kernel_size // 2
        channels = pixels.shape[1]
        kernel_1d = self._gaussian_1d(kernel_size, device=pixels.device, dtype=pixels.dtype)

        horizontal = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        vertical = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)

        x = F.pad(pixels, (pad, pad, 0, 0), mode="reflect")
        x = F.conv2d(x, horizontal, groups=channels)
        x = F.pad(x, (0, 0, pad, pad), mode="reflect")
        return F.conv2d(x, vertical, groups=channels)

    def _gaussian_1d(self, kernel_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(kernel_size), str(device), dtype)
        cached = self._gaussian_kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel = torch.as_tensor(cv2.getGaussianKernel(kernel_size, 0).reshape(-1), device=device, dtype=dtype)
        self._gaussian_kernel_cache[key] = kernel
        return kernel

    def encode_pixels(self, pixels_01: torch.Tensor) -> torch.Tensor:
        mean = self.encoder.image_mean.to(device=pixels_01.device, dtype=pixels_01.dtype)
        std = self.encoder.image_std.to(device=pixels_01.device, dtype=pixels_01.dtype)
        pixel_values = (pixels_01 - mean) / std
        return self._encode_normalized_pixels(pixel_values).cpu()

    def encode_kernel_stack(self, kernel_image_stack: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
        features = [
            self.encode_pixels(kernel_image_stack[kernel_index]).to(dtype=storage_dtype)
            for kernel_index in range(kernel_image_stack.shape[0])
        ]
        return torch.stack(features, dim=1)

    def _encode_normalized_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        from huper.core.mqt_encoder import infer_tgt_size

        vision_device = self.encoder.vision_tower.device
        pixel_values = pixel_values.to(device=vision_device, dtype=self.encoder.vision_tower.dtype)
        vision_features = self._vision_tower_forward(self.encoder.vision_tower, pixel_values)

        abstractor_device = next(self.encoder.query_abstractor.parameters()).device
        vision_features = vision_features.to(abstractor_device)
        tgt_size = self.encoder.tgt_size if self.encoder.tgt_size is not None else infer_tgt_size(vision_features.shape[1])
        tokens = self._query_abstractor_pre_proj(
            self.encoder.query_abstractor,
            vision_features,
            num_visual_tokens="256",
            tgt_size=tgt_size,
        )
        tokens = tokens[:, : self.encoder.num_tokens]
        return tokens.detach()

    @staticmethod
    def _vision_tower_forward(vision_tower, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run CLIP vision tower without the LLaVA wrapper's ``@torch.no_grad()``."""

        outputs = vision_tower.vision_tower(
            pixel_values.to(device=vision_tower.device, dtype=vision_tower.dtype),
            output_hidden_states=True,
        )
        return vision_tower.feature_select(outputs).to(pixel_values.dtype)

    @staticmethod
    def _query_abstractor_pre_proj(query_abstractor, image_features, num_visual_tokens, tgt_size) -> torch.Tensor:
        """Run the MQT query abstractor and return pre-projection tokens."""

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

        return query_abstractor.attn(
            query_abstractor.ln_q(query + query_pos),
            query_abstractor.ln_k(query_abstractor._repeat(pos_embed, batch_size)),
            query_abstractor.ln_v(x),
            attn_mask=None,
        )[0].permute(1, 0, 2)
