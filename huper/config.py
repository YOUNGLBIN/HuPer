"""Configuration loading and pinned Hugging Face asset resolution."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from huper.core.paths import PROJECT_DIR, resolve_hf_model_path


REQUIRED_SECTIONS = {
    "paths",
    "models",
    "experiment",
    "preprocess",
    "train",
    "inference",
    "evaluation",
}

PATH_FIELDS = (
    ("paths", "nsd_root"),
    ("paths", "output_root"),
    ("paths", "hf_home"),
    ("paths", "spice_home"),
    ("paths", "references"),
    ("preprocess", "output_path"),
    ("train", "output_dir"),
    ("train", "feature_cache"),
    ("inference", "checkpoint"),
    ("inference", "output_dir"),
    ("evaluation", "input_root"),
    ("evaluation", "output_dir"),
)

PAPER_PROMPT = """<image>
Task: Describe the image as simply as possible.

Objective: Provide a neutral, one-sentence description of the image. Focus only on the primary subject and its immediate action or position.

Strict Rules:
1. Word Count: Must be strictly between 7 and 11 words.
2. No Prefixes: Do not use "A photo of", "The image shows", or "There is".
3. Start Directly: Begin the sentence immediately with the main subject.

Output (Strictly 7-11 words):"""


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    missing = sorted(REQUIRED_SECTIONS - set(raw))
    if missing:
        raise KeyError(f"Missing configuration section(s): {', '.join(missing)}")

    config = deepcopy(raw)
    for section, key in PATH_FIELDS:
        value = config[section].get(key)
        if value not in {None, ""}:
            config[section][key] = str(_project_path(value))

    local_model_path = config["models"].get("mqt_model_path")
    if local_model_path:
        config["models"]["mqt_model_path"] = str(_project_path(local_model_path))

    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_DIR)
    validate_paper_config(config)
    return config


def validate_paper_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    subjects = [int(subject) for subject in experiment["subjects"]]
    if subjects != [1, 2, 5, 7]:
        raise ValueError("The paper configuration requires subjects [1, 2, 5, 7].")

    kernels = [int(kernel) for kernel in experiment["kernels"]]
    if kernels != [1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63]:
        raise ValueError("The paper configuration requires the fixed 12-kernel Gaussian bank.")
    if int(experiment["mqt_num_tokens"]) != 256:
        raise ValueError("The paper configuration requires 256 MQT tokens.")
    if int(experiment["seed"]) != 42:
        raise ValueError("The formal HuPer paper configuration uses seed 42.")

    training = config["train"]
    if training["recon_loss"] != "mse":
        raise ValueError("The formal HuPer paper configuration uses MSE reconstruction loss.")
    if int(training["num_epochs"]) != 160:
        raise ValueError("The formal HuPer paper configuration uses 160 epochs.")
    if float(training["fusion_prior_alpha"]) != 1.0:
        raise ValueError("The formal HuPer paper configuration uses fusion_prior_alpha=1.")
    if float(training["fusion_kl_weight"]) != 0.1:
        raise ValueError("The formal HuPer paper configuration uses fusion_kl_weight=0.1.")
    expected_training = {
        "batch_size": 32,
        "val_batch_size": 64,
        "max_lr": 0.0003,
        "use_norm": False,
        "use_token": True,
        "eval_val": False,
        "fusion_init_std": 0.01,
    }
    for key, expected in expected_training.items():
        if training[key] != expected:
            raise ValueError(f"The formal HuPer paper configuration requires train.{key}={expected!r}.")

    preprocessing = config["preprocess"]
    if list(preprocessing["splits"]) != ["train", "val", "test"]:
        raise ValueError("The formal cache must cover train, val, and test splits.")
    if preprocessing["mqt_model_dtype"] != "float16" or preprocessing["storage_dtype"] != "float16":
        raise ValueError("The formal cache uses float16 MQT inference and float16 storage.")

    inference = config["inference"]
    if inference["prompt"].strip() != PAPER_PROMPT:
        raise ValueError("The formal HuPer paper configuration requires the fixed 7-11 word prompt.")
    if float(inference["temperature"]) != 0.0:
        raise ValueError("The formal HuPer paper configuration uses temperature=0.")
    if int(inference["num_beams"]) != 1 or int(inference["max_new_tokens"]) != 128:
        raise ValueError("The formal HuPer paper configuration uses one beam and max_new_tokens=128.")


def require_nsd_root(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["nsd_root"])
    if str(path) == "/path/to/nsd" or not path.is_dir():
        config_path = config.get("_config_path", "configs/local.yaml")
        raise FileNotFoundError(
            f"Set paths.nsd_root in {config_path} to the NSD directory "
            "containing webdataset_avg_split/."
        )
    return path


def prepare_runtime_environment(config: dict[str, Any], *, offline: bool | None = None) -> None:
    hf_home = Path(config["paths"]["hf_home"])
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")

    use_offline = bool(config["models"].get("offline", True)) if offline is None else bool(offline)
    os.environ["HF_HUB_OFFLINE"] = "1" if use_offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if use_offline else "0"


def resolve_model_snapshot(
    config: dict[str, Any],
    *,
    repo_key: str,
    revision_key: str,
    allow_download: bool = False,
) -> Path:
    models = config["models"]
    if repo_key == "mqt_repo_id" and models.get("mqt_model_path"):
        return resolve_hf_model_path(models["mqt_model_path"])

    from huggingface_hub import snapshot_download

    repo_id = str(models[repo_key])
    revision = str(models[revision_key])
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(Path(config["paths"]["hf_home"]) / "hub"),
            local_files_only=not allow_download,
        )
    except Exception as exc:
        action = "download" if allow_download else "find in the local cache"
        config_path = config.get("_config_path", "configs/local.yaml")
        raise RuntimeError(
            f"Could not {action} {repo_id}@{revision}. "
            f"Run `huper download-models --config {config_path}` first."
        ) from exc
    return Path(snapshot).resolve()


def resolve_mqt_snapshot(config: dict[str, Any], *, allow_download: bool = False) -> Path:
    return resolve_model_snapshot(
        config,
        repo_key="mqt_repo_id",
        revision_key="mqt_revision",
        allow_download=allow_download,
    )


def resolve_clip_snapshot(config: dict[str, Any], *, allow_download: bool = False) -> Path:
    return resolve_model_snapshot(
        config,
        repo_key="clip_repo_id",
        revision_key="clip_revision",
        allow_download=allow_download,
    )
