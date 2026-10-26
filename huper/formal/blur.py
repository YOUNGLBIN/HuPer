#!/usr/bin/env python3
"""Gaussian blur kernel configuration for the formal experiment."""

from __future__ import annotations


DEFAULT_KERNELS = (1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63)


def validate_gaussian_kernels(kernels: list[int] | tuple[int, ...]) -> None:
    if not kernels:
        raise ValueError("Gaussian kernel list must not be empty.")
    for kernel in kernels:
        if int(kernel) <= 0 or int(kernel) % 2 == 0:
            raise ValueError(f"All Gaussian kernels must be positive odd integers, got {kernel}.")
