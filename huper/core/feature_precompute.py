#!/usr/bin/env python3
"""Shared unique-image feature precomputation loop."""

from __future__ import annotations

import torch
from tqdm import tqdm

from huper.core.nsd_data import coco_id_to_int, unique_image_dataset


def encode_unique_images(
    *,
    shards: list[str],
    batch_size: int,
    encode_batch,
    storage_dtype: torch.dtype,
    progress_desc: str,
    max_images: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    features_by_id: dict[int, torch.Tensor] = {}
    pending_ids: list[int] = []
    pending_images: list[torch.Tensor] = []
    progress = tqdm(desc=progress_desc, unit="img")

    def flush() -> int:
        if not pending_images:
            return 0
        images = torch.stack(pending_images, dim=0)
        with torch.inference_mode():
            features = encode_batch(images).detach().to("cpu", dtype=storage_dtype)
        for coco_id, feature in zip(pending_ids, features):
            features_by_id[coco_id] = feature
        count = len(pending_images)
        pending_ids.clear()
        pending_images.clear()
        return count

    for coco_id, image in unique_image_dataset(shards):
        if max_images is not None and len(features_by_id) + len(pending_ids) >= max_images:
            break
        coco_int = coco_id_to_int(coco_id)
        if coco_int in features_by_id or coco_int in pending_ids:
            continue

        pending_ids.append(coco_int)
        pending_images.append(image)
        if len(pending_images) >= batch_size:
            progress.update(flush())

    progress.update(flush())
    progress.close()

    if not features_by_id:
        raise RuntimeError("No unique images were found in the selected NSD shards.")

    ids = torch.tensor(sorted(features_by_id), dtype=torch.long)
    features = torch.stack([features_by_id[int(coco_id)] for coco_id in ids.tolist()], dim=0)
    return ids, features
