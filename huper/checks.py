"""Validate the external assets and runtime needed by HuPer."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from huper.config import require_nsd_root, resolve_clip_snapshot, resolve_mqt_snapshot
from huper.core.nsd_data import collect_nsd_shards
from huper.core.paths import save_json
from huper.spice_assets import resolve_corenlp_assets


PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "webdataset",
    "pycocoevalcap",
    "opencv-python",
)


def _java_version() -> str:
    try:
        result = subprocess.run(
            ["java", "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Java 11+ is required by METEOR and SPICE evaluation.") from exc
    first_line = (result.stderr or result.stdout).splitlines()[0]
    return first_line


def run_checks(config: dict) -> Path:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"HuPer requires Python 3.10, found {sys.version.split()[0]}.")

    nsd_root = require_nsd_root(config)
    subjects = [int(subject) for subject in config["experiment"]["subjects"]]
    splits = list(config["preprocess"]["splits"])
    shards = collect_nsd_shards(nsd_root, subjects, splits)
    if len(shards) != 84:
        raise RuntimeError(f"Expected 84 NSD shards for the paper split, found {len(shards)}.")

    reference_path = Path(config["paths"]["references"])
    with reference_path.open("r", encoding="utf-8") as handle:
        references = json.load(handle)
    if len(references) != 982:
        raise RuntimeError(f"Expected 982 BrainHub reference entries, found {len(references)}.")

    mqt_path = resolve_mqt_snapshot(config)
    clip_path = resolve_clip_snapshot(config)
    corenlp_paths = resolve_corenlp_assets(config)
    java_version = _java_version()
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}

    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root)
    report = {
        "status": "ok",
        "python": sys.version.split()[0],
        "packages": versions,
        "java": java_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nsd_root": str(nsd_root),
        "num_nsd_shards": len(shards),
        "num_references": len(references),
        "mqt_snapshot": str(mqt_path),
        "clip_snapshot": str(clip_path),
        "spice_corenlp_jars": {name: str(path) for name, path in corenlp_paths.items()},
        "output_root": str(output_root),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "cache_disk_recommendation_gib": 230,
    }
    report_path = output_root / "check_report.json"
    save_json(report_path, report)

    print(f"Environment check passed; report: {report_path}")
    if report["disk_free_gib"] < report["cache_disk_recommendation_gib"]:
        print(
            f"WARNING: only {report['disk_free_gib']} GiB free; "
            "the complete Gaussian cache needs approximately 214 GiB."
        )
    return report_path
