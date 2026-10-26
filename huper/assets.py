"""Download the exact model revisions used by the HuPer paper pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path

from huper.config import (
    prepare_runtime_environment,
    resolve_clip_snapshot,
    resolve_mqt_snapshot,
)
from huper.core.paths import PROJECT_DIR, save_json
from huper.spice_assets import (
    CORENLP_JARS,
    CORENLP_URLS,
    CORENLP_VERSION,
    prepare_corenlp_assets,
)


def download_models(config: dict) -> Path:
    prepare_runtime_environment(config, offline=False)
    mqt_path = resolve_mqt_snapshot(config, allow_download=True)
    clip_path = resolve_clip_snapshot(config, allow_download=True)
    corenlp_paths = prepare_corenlp_assets(config)
    manifest_path = PROJECT_DIR / "models" / "download_manifest.json"
    save_json(
        manifest_path,
        {
            "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "mqt": {
                "repo_id": config["models"]["mqt_repo_id"],
                "revision": config["models"]["mqt_revision"],
                "snapshot_path": str(mqt_path),
            },
            "clip": {
                "repo_id": config["models"]["clip_repo_id"],
                "revision": config["models"]["clip_revision"],
                "snapshot_path": str(clip_path),
            },
            "spice": {
                "corenlp_version": CORENLP_VERSION,
                "source_urls": list(CORENLP_URLS),
                "jars": {
                    name: {
                        "path": str(path),
                        "size": CORENLP_JARS[name]["size"],
                        "sha256": CORENLP_JARS[name]["sha256"],
                    }
                    for name, path in corenlp_paths.items()
                },
            },
        },
    )
    print(f"MQT-LLaVA snapshot: {mqt_path}")
    print(f"CLIP-336 snapshot: {clip_path}")
    print(f"SPICE CoreNLP assets: {next(iter(corenlp_paths.values())).parent}")
    print(f"Download manifest: {manifest_path}")
    return manifest_path
