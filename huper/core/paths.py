#!/usr/bin/env python3
"""Repository-local path, model-resolution, and JSON helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent

DEFAULT_DATA_PATH = Path(os.environ.get("HUPER_DATA_PATH", PROJECT_DIR / "data" / "nsd"))
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "train"
DEFAULT_FEATURE_CACHE_DIR = PROJECT_DIR / "outputs" / "cache"
DEFAULT_FORMAL_CACHE_DIR = DEFAULT_FEATURE_CACHE_DIR
DEFAULT_MQT_REPO_PATH = PROJECT_DIR
DEFAULT_MQT_MODEL_PATH = os.environ.get(
    "HUPER_MQT_MODEL_PATH",
    str(PROJECT_DIR / "models" / "MQT-LLaVA-7b"),
)


def add_repo_to_path() -> None:
    path_str = str(PROJECT_DIR.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def setup_import_paths(mqt_repo_path: str | Path | None = None) -> None:
    add_repo_to_path()
    add_mqt_repo_to_path(mqt_repo_path or DEFAULT_MQT_REPO_PATH)


def add_mqt_repo_to_path(repo_path: str | Path = DEFAULT_MQT_REPO_PATH) -> Path:
    path = Path(repo_path).expanduser().resolve()
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def resolve_hf_model_path(model_path: str | Path) -> Path:
    """Resolve a local Hugging Face repository or snapshot directory."""

    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    if (path / "config.json").exists():
        return path.resolve()

    ref_path = path / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = path / "snapshots" / revision
        if snapshot_path.exists():
            return snapshot_path.resolve()
        raise FileNotFoundError(f"Snapshot referenced by {ref_path} is missing: {snapshot_path}")

    snapshots_dir = path / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir())
        if len(snapshots) == 1:
            return snapshots[0].resolve()

    raise FileNotFoundError(
        f"{path} is not a loadable model directory. Expected config.json or one cached snapshot."
    )


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)
