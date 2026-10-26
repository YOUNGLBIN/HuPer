#!/usr/bin/env python3
"""COCO-id keyed tensor caches used by HuPer training."""

from __future__ import annotations

from pathlib import Path

import torch


class CocoTensorCache:
    """Memory-mapped tensor cache keyed by ``coco73k.npy`` ids."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_ndim: int,
        device: torch.device,
        pin_memory: bool = False,
    ):
        self.path = Path(path).expanduser()
        payload = torch.load(self.path, map_location="cpu", mmap=True)
        # payload = torch.load(self.path, map_location="cpu")
        self.ids = payload["ids"].long().view(-1)
        self.features = payload["features"].detach()
        self.metadata = payload.get("metadata", {})
        self.device = torch.device(device)

        if self.features.ndim != expected_ndim:
            raise ValueError(f"Expected features with ndim={expected_ndim}, got {tuple(self.features.shape)}.")
        if self.ids.numel() != self.features.shape[0]:
            raise ValueError(f"Cache ids/features length mismatch: {self.ids.numel()} vs {self.features.shape[0]}")

        self.features = self.features.to(self.device)
        if pin_memory and self.device.type == "cpu":
            self.features = self.features.pin_memory()

        max_id = int(self.ids.max().item()) if self.ids.numel() else -1
        self.row_for_coco = torch.full((max_id + 1,), -1, dtype=torch.long, device=self.device)
        self.row_for_coco[self.ids.to(self.device)] = torch.arange(
            self.ids.numel(),
            dtype=torch.long,
            device=self.device,
        )

    def get(self, coco_ids, *, device: torch.device) -> torch.Tensor:
        ids = torch.as_tensor(coco_ids, dtype=torch.long, device=self.device).view(-1)
        if ids.numel() == 0:
            raise ValueError("Empty coco id batch.")
        if int(ids.max().item()) >= self.row_for_coco.numel():
            raise KeyError(f"coco id {int(ids.max().item())} is outside cache index range for {self.path}")

        rows = self.row_for_coco.index_select(0, ids)
        if (rows < 0).any():
            missing = ids[rows < 0][:8].detach().cpu().tolist()
            raise KeyError(f"Missing coco id(s) in cache {self.path}: {missing}")
        return self.features.index_select(0, rows).to(device=device, non_blocking=True)
