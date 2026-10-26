"""Project-local Stanford CoreNLP assets required by SPICE evaluation."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from huper.core.paths import PROJECT_DIR


CORENLP_VERSION = "3.6.0"
CORENLP_ARCHIVE_NAME = "stanford-corenlp-full-2015-12-09"
CORENLP_URLS = (
    f"https://nlp.stanford.edu/software/{CORENLP_ARCHIVE_NAME}.zip",
    f"http://nlp.stanford.edu/software/{CORENLP_ARCHIVE_NAME}.zip",
)
CORENLP_JARS = {
    f"stanford-corenlp-{CORENLP_VERSION}.jar": {
        "size": 7_270_741,
        "sha256": "d341d5863276f55782c5ef73212222d6b3b48e02ae1951ab6de7618d28b12742",
    },
    f"stanford-corenlp-{CORENLP_VERSION}-models.jar": {
        "size": 378_116_526,
        "sha256": "d285157bd7887ee3018094b5dbae092e49088b023361ccdb550ee1daf9b2c9bd",
    },
}


def spice_home(config: dict | None = None) -> Path:
    configured = None if config is None else config.get("paths", {}).get("spice_home")
    return Path(configured or PROJECT_DIR / "models" / "spice").expanduser().resolve()


def corenlp_lib_dir(config: dict | None = None) -> Path:
    return spice_home(config) / "lib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_jar(path: Path, specification: dict[str, int | str]) -> str | None:
    if not path.is_file():
        return "missing"
    expected_size = int(specification["size"])
    if path.stat().st_size != expected_size:
        return f"size {path.stat().st_size}, expected {expected_size}"
    actual_hash = _sha256(path)
    if actual_hash != specification["sha256"]:
        return f"SHA256 {actual_hash}, expected {specification['sha256']}"
    return None


def resolve_corenlp_assets(config: dict | None = None) -> dict[str, Path]:
    """Return verified project-local CoreNLP jars or raise an actionable error."""

    lib_dir = corenlp_lib_dir(config)
    paths = {name: lib_dir / name for name in CORENLP_JARS}
    problems = {
        name: problem
        for name, path in paths.items()
        if (problem := _validate_jar(path, CORENLP_JARS[name])) is not None
    }
    if problems:
        details = "; ".join(f"{name}: {problem}" for name, problem in problems.items())
        config_path = "configs/local.yaml"
        if config is not None:
            config_path = str(config.get("_config_path", config_path))
        raise RuntimeError(
            f"Stanford CoreNLP {CORENLP_VERSION} assets required by SPICE are not ready in {lib_dir}: "
            f"{details}. Run `huper download-models --config {config_path}`."
        )
    return paths


def _copy_installed_assets(destination: Path) -> bool:
    """Reuse valid jars from the active pycocoevalcap installation when present."""

    from pycocoevalcap.spice import spice as spice_module

    source_dir = Path(spice_module.__file__).resolve().parent / "lib"
    for name, specification in CORENLP_JARS.items():
        if _validate_jar(source_dir / name, specification) is not None:
            return False

    destination.mkdir(parents=True, exist_ok=True)
    for name in CORENLP_JARS:
        temporary_path = destination / f"{name}.tmp"
        shutil.copy2(source_dir / name, temporary_path)
        temporary_path.replace(destination / name)
    print(f"Copied Stanford CoreNLP {CORENLP_VERSION} from the active pycocoevalcap installation.")
    return True


def _download_with_resume(destination: Path, *, retries: int = 5, timeout: int = 120) -> None:
    part_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for url in CORENLP_URLS:
        for attempt in range(1, retries + 1):
            offset = part_path.stat().st_size if part_path.exists() else 0
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "HuPer/0.1 CoreNLP asset downloader",
                    **({"Range": f"bytes={offset}-"} if offset else {}),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    append = offset > 0 and getattr(response, "status", None) == 206
                    mode = "ab" if append else "wb"
                    downloaded = offset if append else 0
                    next_report = downloaded + 64 * 1024 * 1024
                    with part_path.open(mode) as handle:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if downloaded >= next_report:
                                print(f"CoreNLP download: {downloaded / 1024**2:.0f} MiB")
                                next_report = downloaded + 64 * 1024 * 1024
                part_path.replace(destination)
                return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                print(f"CoreNLP download attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))

    raise RuntimeError(
        f"Could not download Stanford CoreNLP {CORENLP_VERSION} from the official Stanford NLP server. "
        f"The partial download, if any, remains at {part_path} and will be resumed next time."
    ) from last_error


def _extract_verified_jars(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name, specification in CORENLP_JARS.items():
                member = f"{CORENLP_ARCHIVE_NAME}/{name}"
                temporary_path = destination / f"{name}.tmp"
                with archive.open(member) as source, temporary_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                problem = _validate_jar(temporary_path, specification)
                if problem is not None:
                    temporary_path.unlink(missing_ok=True)
                    raise RuntimeError(f"Downloaded {name} failed integrity validation: {problem}.")
                os.replace(temporary_path, destination / name)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Downloaded CoreNLP archive is incomplete or invalid: {archive_path}. "
            "Remove it and rerun `huper download-models`."
        ) from exc


def prepare_corenlp_assets(config: dict) -> dict[str, Path]:
    """Install and verify CoreNLP jars under the configured HuPer model directory."""

    try:
        return resolve_corenlp_assets(config)
    except RuntimeError:
        pass

    destination = corenlp_lib_dir(config)
    if _copy_installed_assets(destination):
        return resolve_corenlp_assets(config)

    archive_path = spice_home(config) / "downloads" / f"{CORENLP_ARCHIVE_NAME}.zip"
    print(
        f"Downloading Stanford CoreNLP {CORENLP_VERSION} for SPICE "
        f"(approximately 370 MiB installed) ..."
    )
    if not archive_path.is_file():
        _download_with_resume(archive_path)
    _extract_verified_jars(archive_path, destination)
    archive_path.unlink(missing_ok=True)
    print(f"Stanford CoreNLP assets: {destination}")
    return resolve_corenlp_assets(config)
