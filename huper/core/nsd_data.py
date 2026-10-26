#!/usr/bin/env python3
"""NSD WebDataset, COCO-id, and scheduler helpers for HuPer."""

from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import braceexpand
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader


def nsd_split_shards(data_path: str | Path, subject: int, split: str) -> list[str]:
    base = Path(data_path).expanduser() / "webdataset_avg_split"
    if split == "train":
        pattern = base / "train" / f"train_subj0{subject}_{{0..17}}.tar"
    elif split == "val":
        pattern = base / "val" / f"val_subj0{subject}_0.tar"
    elif split == "test":
        pattern = base / "test" / f"test_subj0{subject}_{{0..1}}.tar"
    else:
        raise ValueError(f"Unsupported split: {split}")
    return list(braceexpand.braceexpand(str(pattern)))


def collect_nsd_shards(data_path: str | Path, subjects: list[int], splits: list[str]) -> list[str]:
    shards: list[str] = []
    for subject in [int(sub) for sub in subjects]:
        for split in splits:
            shards.extend(nsd_split_shards(data_path, subject, split))

    missing = [path for path in shards if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing WebDataset shard(s):\n" + "\n".join(missing[:8]))
    return shards


def coco_id_to_int(coco_id) -> int:
    return int(np.asarray(coco_id).reshape(-1)[0])


def unique_image_dataset(shards: list[str]):
    return (
        wds.WebDataset(shards, resampled=False, shardshuffle=False)
        .decode("torch")
        .rename(images="jpg;png", coco="coco73k.npy")
        .to_tuple("coco", "images")
    )


def load_npy_from_sample(sample: dict, key: str):
    return np.load(io.BytesIO(sample[key]))


class VoxelCocoDecoder:
    def __init__(self, voxels_key: str):
        self.voxels_key = voxels_key

    def __call__(self, sample: dict):
        voxels = load_npy_from_sample(sample, self.voxels_key)
        coco = load_npy_from_sample(sample, "coco73k.npy")
        return voxels, coco


def voxel_coco_webdataset(
    urls,
    *,
    voxels_key: str,
    cache_dir: str,
    resampled: bool,
    nodesplitter,
):
    return wds.WebDataset(urls, resampled=resampled, cache_dir=cache_dir, nodesplitter=nodesplitter).map(
        VoxelCocoDecoder(voxels_key)
    )


def make_brainx_dataloaders(
    batch_size,
    *,
    num_devices,
    num_workers,
    train_url,
    val_url,
    meta_url,
    num_train,
    num_val,
    cache_dir,
    voxels_key,
    val_batch_size,
):
    def my_split_by_node(urls):
        return urls

    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))

    if not Path(train_url[0]).exists():
        raise FileNotFoundError(f"Training shard not found: {train_url[0]}")
    if not Path(val_url[0]).exists():
        raise FileNotFoundError(f"Validation shard not found: {val_url[0]}")

    if num_train is None:
        metadata = json.load(open(meta_url, "r", encoding="utf-8"))
        num_train = metadata["totals"]["train"]
    if num_val is None:
        metadata = json.load(open(meta_url, "r", encoding="utf-8"))
        num_val = metadata["totals"]["val"]

    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0:
        num_worker_batches = 1

    print("\nnum_train", num_train)
    print("global_batch_size", global_batch_size)
    print("batch_size", batch_size)
    print("num_workers", num_workers)
    print("num_batches", num_batches)
    print("num_worker_batches", num_worker_batches)

    train_data = (
        voxel_coco_webdataset(
            train_url,
            voxels_key=voxels_key,
            cache_dir=cache_dir,
            resampled=True,
            nodesplitter=my_split_by_node,
        )
        .shuffle(500, initial=500, rng=random.Random(42))
        .slice(num_train)
        .batched(batch_size, partial=True)
        .with_epoch(num_worker_batches)
    )
    train_dl = DataLoader(train_data, batch_size=None, num_workers=1, shuffle=False)

    num_batches = math.floor(num_val / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0:
        num_worker_batches = 1

    print("\nnum_val", num_val)
    print("val_num_batches", num_batches)
    print("val_batch_size", val_batch_size)

    val_data = voxel_coco_webdataset(
        val_url,
        voxels_key=voxels_key,
        cache_dir=cache_dir,
        resampled=False,
        nodesplitter=my_split_by_node,
    ).batched(val_batch_size, partial=False)
    val_dl = DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)

    return train_dl, val_dl, num_train, num_val


def prepare_brainx_dataloaders(args, subs, batch_size, num_devices, num_workers, print_fn):
    num_train = 8559 + 300
    num_val = 982
    train_dls = []
    val_dls = []
    for sub in subs:
        train_url = (
            "{"
            + f"{args.data_path}/webdataset_avg_split/train/train_subj0{sub}_"
            + "{0..17}.tar,"
            + f"{args.data_path}/webdataset_avg_split/val/val_subj0{sub}_0.tar"
            + "}"
        )
        val_url = f"{args.data_path}/webdataset_avg_split/test/test_subj0{sub}_" + "{0..1}.tar"
        meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{sub}.json"
        print_fn(train_url, "\n", val_url)
        train_dl, val_dl, num_train, num_val = make_brainx_dataloaders(
            batch_size,
            num_devices=num_devices,
            num_workers=num_workers,
            train_url=train_url,
            val_url=val_url,
            meta_url=meta_url,
            num_train=num_train,
            num_val=num_val,
            val_batch_size=args.val_batch_size,
            cache_dir=args.data_path,
            voxels_key="nsdgeneral.npy",
        )
        train_dls.append(train_dl)
        val_dls.append((sub, val_dl))
    return train_dls, val_dls, num_train


def build_lr_scheduler(args, optimizer, num_train, batch_size, num_devices, num_epochs):
    global_batch_size = batch_size * num_devices
    total_steps = int(num_epochs * (num_train // global_batch_size))
    if args.lr_scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps,
            last_epoch=-1,
        )

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / num_epochs,
    )


def make_caption_val_loader(data_path, subject: int, batch_size: int):
    val_url = [
        f"{data_path}/webdataset_avg_split/test/test_subj0{subject}_{idx}.tar"
        for idx in range(2)
    ]
    val_data = (
        wds.WebDataset(
            val_url,
            resampled=False,
            cache_dir=data_path,
            nodesplitter=lambda urls: urls,
            shardshuffle=False,
        )
        .decode("torch")
        .rename(
            images="jpg;png",
            voxels="nsdgeneral.npy",
            trial="trial.npy",
            coco="coco73k.npy",
            reps="num_uniques.npy",
        )
        .to_tuple("voxels", "images")
        .batched(batch_size, partial=False)
    )
    return torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)


def make_caption_image_loader(data_path, subject: int, batch_size: int):
    """Load only NSD test images, preserving the caption-loader sample order."""

    test_urls = nsd_split_shards(data_path, subject, "test")
    missing = [path for path in test_urls if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing WebDataset shard(s):\n" + "\n".join(missing))

    image_data = (
        wds.WebDataset(
            test_urls,
            resampled=False,
            cache_dir=str(Path(data_path).expanduser()),
            nodesplitter=lambda urls: urls,
            shardshuffle=False,
        )
        .decode("torch")
        .rename(images="jpg;png")
        .to_tuple("images")
        .batched(batch_size, partial=True)
    )
    return torch.utils.data.DataLoader(image_data, batch_size=None, num_workers=0, shuffle=False)
