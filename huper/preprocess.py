"""Precompute the formal multi-kernel Gaussian MQT feature cache."""

from __future__ import annotations

import datetime
from pathlib import Path

import torch

from huper.config import require_nsd_root, resolve_clip_snapshot, resolve_mqt_snapshot
from huper.core.feature_precompute import encode_unique_images
from huper.core.nsd_data import collect_nsd_shards
from huper.core.paths import save_json
from huper.formal.blur import validate_gaussian_kernels
from huper.formal.gaussian_mqt_encoder import MultiKernelGaussianMQTEncoder


def _metadata(
    config,
    *,
    ids: torch.Tensor,
    features: torch.Tensor,
    shards: list[str],
    encoder,
    mqt_path: Path,
    clip_path: Path,
    max_images: int | None,
) -> dict:
    experiment = config["experiment"]
    preprocessing = config["preprocess"]
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cache_type": "multi_kernel_gaussian_mqt",
        "feature_space": "mqt",
        "encoder_module": "huper.formal.gaussian_mqt_encoder",
        "use_image_aug": False,
        "blur_mode": "gaussian",
        "kernels": [int(kernel) for kernel in experiment["kernels"]],
        "mqt_repo_id": config["models"]["mqt_repo_id"],
        "mqt_revision": config["models"]["mqt_revision"],
        "mqt_model_path": str(mqt_path),
        "clip_repo_id": config["models"]["clip_repo_id"],
        "clip_revision": config["models"]["clip_revision"],
        "clip_model_path": str(clip_path),
        "mqt_num_tokens": int(experiment["mqt_num_tokens"]),
        "mqt_model_dtype": preprocessing["mqt_model_dtype"],
        "mqt_tgt_size": None,
        "image_size": list(encoder.image_size),
        "storage_dtype": str(features.dtype).replace("torch.", ""),
        "num_images": int(ids.numel()),
        "feature_shape": list(features.shape),
        "feature_shape_meaning": ["image", "kernel", "token", "dim"],
        "id_key": "coco73k.npy",
        "ids_dtype": str(ids.dtype).replace("torch.", ""),
        "data_path": str(config["paths"]["nsd_root"]),
        "subjects": [int(subject) for subject in experiment["subjects"]],
        "splits": list(preprocessing["splits"]),
        "num_shards": len(shards),
        "complete": max_images is None,
        "max_images": max_images,
    }


def run_preprocess(
    config: dict,
    *,
    overwrite: bool = False,
    max_images: int | None = None,
) -> Path:
    """Build the cache and return its path."""

    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be at least 1.")

    nsd_root = require_nsd_root(config)
    experiment = config["experiment"]
    preprocessing = config["preprocess"]
    kernels = [int(kernel) for kernel in experiment["kernels"]]
    validate_gaussian_kernels(kernels)

    output_path = Path(preprocessing["output_path"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mqt_path = resolve_mqt_snapshot(config)
    clip_path = resolve_clip_snapshot(config)
    subjects = [int(subject) for subject in experiment["subjects"]]
    splits = list(preprocessing["splits"])
    shards = collect_nsd_shards(nsd_root, subjects, splits)
    storage_dtype = getattr(torch, preprocessing["storage_dtype"])

    print(f"Loading MQT encoder on {preprocessing['device']}...")
    encoder = MultiKernelGaussianMQTEncoder(
        kernels=kernels,
        device=preprocessing["device"],
        mqt_repo_path=config["_project_root"],
        mqt_model_path=str(mqt_path),
        vision_model_path=str(clip_path),
        mqt_num_tokens=int(experiment["mqt_num_tokens"]),
        mqt_model_dtype=preprocessing["mqt_model_dtype"],
        local_files_only=True,
    )
    print(
        f"tokens={encoder.num_tokens}, dim={encoder.token_dim}, "
        f"kernels={kernels}, image_size={encoder.image_size}"
    )

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        prepared = encoder.prepare_pixels(images)
        blur_stack = encoder.gaussian_blur_stack(prepared)
        return encoder.encode_kernel_stack(blur_stack, storage_dtype=storage_dtype)

    ids, features = encode_unique_images(
        shards=shards,
        batch_size=int(preprocessing["batch_size"]),
        encode_batch=encode_batch,
        storage_dtype=storage_dtype,
        progress_desc="encoding unique multi-kernel images",
        max_images=max_images,
    )
    metadata = _metadata(
        config,
        ids=ids,
        features=features,
        shards=shards,
        encoder=encoder,
        mqt_path=mqt_path,
        clip_path=clip_path,
        max_images=max_images,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save({"ids": ids, "features": features, "metadata": metadata}, temporary_path)
    temporary_path.replace(output_path)
    save_json(output_path.with_suffix(".json"), metadata)
    print(f"Saved {features.shape[0]} image features {tuple(features.shape)} to {output_path}")
    return output_path
