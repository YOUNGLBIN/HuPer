#!/usr/bin/env python3
"""Precomputed multi-kernel Gaussian MQT feature cache."""

from __future__ import annotations

from huper.core.tensor_cache import CocoTensorCache


class MultiKernelMQTFeatureCache(CocoTensorCache):
    """Feature cache with tensors shaped [image, kernel, token, dim]."""

    def __init__(self, path, *, device, pin_memory: bool = False):
        super().__init__(path, expected_ndim=4, device=device, pin_memory=pin_memory)

    @property
    def num_kernels(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.features.shape[2])

    @property
    def feat_dim(self) -> int:
        return int(self.features.shape[3])


def validate_multi_kernel_cache(args, cache: MultiKernelMQTFeatureCache) -> None:
    metadata = cache.metadata
    if metadata.get("feature_space") != "mqt":
        raise ValueError(f"Expected MQT cache, got feature_space={metadata.get('feature_space')!r}.")
    if metadata.get("cache_type") != "multi_kernel_gaussian_mqt":
        raise ValueError(f"Expected multi-kernel Gaussian cache, got cache_type={metadata.get('cache_type')!r}.")
    if metadata.get("use_image_aug") is not False:
        raise ValueError("Formal experiment requires no image augmentation.")
    if metadata.get("complete") is False:
        raise ValueError("The selected cache is a partial smoke-test cache and cannot be used for formal training.")
    if metadata.get("mqt_num_tokens") != args.mqt_num_tokens:
        raise ValueError(f"Cache mqt_num_tokens={metadata.get('mqt_num_tokens')} but args has {args.mqt_num_tokens}.")
    if list(metadata.get("kernels", [])) != list(args.kernels):
        raise ValueError("Cache kernels do not match args.kernels.")
