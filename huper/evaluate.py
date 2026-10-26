"""Self-contained BrainHub COCO-caption evaluation for HuPer outputs."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice import spice as spice_module
from pycocoevalcap.tokenizer import ptbtokenizer

from huper.core.paths import PROJECT_DIR, save_json
from huper.spice_assets import resolve_corenlp_assets


METRIC_NAMES = (
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE",
    "CIDEr",
    "SPICE",
)


def _numeric_keys(keys) -> list[str]:
    try:
        return sorted((str(key) for key in keys), key=int)
    except ValueError as exc:
        raise ValueError("Caption/reference keys must be integer strings.") from exc


def validate_caption_keys(
    references: dict[str, list[str]],
    candidates: dict[str, str],
    *,
    require_complete: bool,
) -> list[str]:
    if not candidates:
        raise ValueError("Candidate caption file is empty.")
    candidate_keys = set(candidates)
    reference_keys = set(references)
    unexpected = _numeric_keys(candidate_keys - reference_keys)
    missing = _numeric_keys(reference_keys - candidate_keys)
    if unexpected:
        raise KeyError(f"Candidate file contains unknown key(s): {unexpected[:8]}")
    if require_complete and missing:
        raise KeyError(
            f"Formal evaluation requires all {len(reference_keys)} captions; "
            f"missing {len(missing)} key(s), beginning with {missing[:8]}."
        )

    keys = _numeric_keys(candidate_keys)
    for key in keys:
        if not isinstance(candidates[key], str) or not candidates[key].strip():
            raise ValueError(f"Candidate caption {key!r} must be a non-empty string.")
        refs = references[key]
        if not isinstance(refs, list) or not refs or not all(isinstance(item, str) for item in refs):
            raise ValueError(f"References for key {key!r} must be a non-empty list of strings.")
    return keys


def _tokenize(references: list[list[str]], candidates: list[str]):
    refs = {
        index: [{"caption": reference} for reference in sample_references]
        for index, sample_references in enumerate(references)
    }
    cands = {
        index: [{"caption": candidate}]
        for index, candidate in enumerate(candidates)
    }
    return _ptb_tokenize_writable(refs), _ptb_tokenize_writable(cands)


def _ptb_tokenize_writable(captions_for_image):
    """Run pycocoevalcap's PTB tokenizer without writing into site-packages."""

    jar_dir = Path(ptbtokenizer.__file__).resolve().parent
    command = [
        "java",
        "-cp",
        ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]
    image_ids = [
        key
        for key, values in captions_for_image.items()
        for _ in range(len(values))
    ]
    sentences = "\n".join(
        caption["caption"].replace("\n", " ")
        for values in captions_for_image.values()
        for caption in values
    )
    with tempfile.NamedTemporaryFile(prefix="huper_ptb_", delete=False) as handle:
        handle.write(sentences.encode())
        temporary_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [*command, str(temporary_path)],
            cwd=jar_dir,
            check=True,
            capture_output=True,
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    lines = completed.stdout.decode().split("\n")
    tokenized = {}
    for key, line in zip(image_ids, lines):
        tokenized.setdefault(key, [])
        tokenized[key].append(
            " ".join(
                word
                for word in line.rstrip().split(" ")
                if word not in ptbtokenizer.PUNCTUATIONS
            )
        )
    return tokenized


def _spice_score_writable(gts, res, *, spice_corenlp_lib: str | Path | None = None) -> float:
    """Run SPICE with temporary/cache files outside installed site-packages."""

    image_ids = sorted(gts)
    input_data = []
    for image_id in image_ids:
        hypothesis = res[image_id]
        references = gts[image_id]
        if not isinstance(hypothesis, list) or len(hypothesis) != 1:
            raise ValueError("SPICE expects one candidate caption per sample.")
        if not isinstance(references, list) or not references:
            raise ValueError("SPICE expects at least one reference caption per sample.")
        input_data.append(
            {
                "image_id": image_id,
                "test": hypothesis[0],
                "refs": references,
            }
        )

    spice_dir = Path(spice_module.__file__).resolve().parent
    spice_jar = spice_dir / spice_module.SPICE_JAR
    package_lib_dir = spice_dir / "lib"
    if not spice_jar.is_file():
        raise RuntimeError(f"pycocoevalcap SPICE runtime jar is missing: {spice_jar}. Reinstall pycocoevalcap==1.2.")
    if spice_corenlp_lib is None:
        verified_assets = resolve_corenlp_assets()
        project_lib_dir = next(iter(verified_assets.values())).parent
    else:
        project_lib_dir = Path(spice_corenlp_lib).resolve()
    classpath = os.pathsep.join(
        (
            str(spice_jar),
            str(package_lib_dir / "*"),
            str(project_lib_dir / "*"),
        )
    )
    cache_dir = Path(
        os.environ.get(
            "HUPER_SPICE_CACHE",
            str(PROJECT_DIR / "outputs" / ".spice_cache"),
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="huper_spice_") as temporary_dir:
        input_path = Path(temporary_dir) / "input.json"
        output_path = Path(temporary_dir) / "output.json"
        input_path.write_text(json.dumps(input_data), encoding="utf-8")
        output_path.touch()
        command = [
            "java",
            "-Xmx8G",
            "-cp",
            classpath,
            "edu.anu.spice.SpiceScorer",
            str(input_path),
            "-cache",
            str(cache_dir),
            "-out",
            str(output_path),
            "-subset",
            "-silent",
        ]
        subprocess.check_call(command, cwd=spice_dir)
        with output_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)

    values = []
    for item in results:
        value = item["scores"]["All"]["f"]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return float(np.mean(np.asarray(values)))


def compute_caption_metrics(
    references: dict[str, list[str]],
    candidates: dict[str, str],
    *,
    require_complete: bool = True,
    spice_corenlp_lib: str | Path | None = None,
) -> dict[str, float]:
    keys = validate_caption_keys(references, candidates, require_complete=require_complete)
    reference_list = [references[key] for key in keys]
    candidate_list = [candidates[key] for key in keys]
    tokenized_refs, tokenized_candidates = _tokenize(reference_list, candidate_list)

    bleu, _ = Bleu(4).compute_score(tokenized_refs, tokenized_candidates)
    meteor, _ = Meteor().compute_score(tokenized_refs, tokenized_candidates)
    rouge, _ = Rouge().compute_score(tokenized_refs, tokenized_candidates)
    cider, _ = Cider().compute_score(tokenized_refs, tokenized_candidates)
    spice = _spice_score_writable(
        tokenized_refs,
        tokenized_candidates,
        spice_corenlp_lib=spice_corenlp_lib,
    )
    values = [*bleu, meteor, rouge, cider, spice]
    return {
        name: float(np.asarray(value).mean())
        for name, value in zip(METRIC_NAMES, values)
    }


def aggregate_subject_metrics(subject_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    if not subject_metrics:
        raise ValueError("No subject metrics were provided.")
    for subject, metrics in subject_metrics.items():
        missing = set(METRIC_NAMES) - set(metrics)
        if missing:
            raise KeyError(f"Subject {subject} is missing metric(s): {sorted(missing)}")
    return {
        metric: float(np.mean([metrics[metric] for metrics in subject_metrics.values()]))
        for metric in METRIC_NAMES
    }


def _subject_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.search(r"sub(?:j)?0?([1257])(?:_|$)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer subject 1/2/5/7 from caption path: {path}")


def discover_caption_files(input_path: str | Path) -> dict[int, Path]:
    path = Path(input_path)
    files = [path] if path.is_file() else sorted(path.rglob("fmricap.json"))
    if not files:
        raise FileNotFoundError(f"No fmricap.json files found under {path}")

    result: dict[int, Path] = {}
    for file_path in files:
        subject = _subject_from_path(file_path)
        if subject in result:
            raise ValueError(
                f"Found multiple caption files for subject {subject}: "
                f"{result[subject]} and {file_path}. Narrow evaluation.input_root."
            )
        result[subject] = file_path
    return result


def _write_metrics_text(path: Path, metrics: dict[str, float], *, candidate_path: Path) -> None:
    lines = [f"Candidates: {candidate_path}", ""]
    lines.extend(f"{name}: {metrics[name]:.4f}" for name in METRIC_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluate(
    config: dict,
    *,
    input_path: str | Path | None = None,
    require_complete: bool | None = None,
) -> Path:
    evaluation = config["evaluation"]
    references_path = Path(config["paths"]["references"])
    with references_path.open("r", encoding="utf-8") as handle:
        references = json.load(handle)

    candidate_files = discover_caption_files(input_path or evaluation["input_root"])
    output_dir = Path(evaluation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    strict = bool(evaluation["require_complete"]) if require_complete is None else require_complete
    corenlp_paths = resolve_corenlp_assets(config)
    spice_corenlp_lib = next(iter(corenlp_paths.values())).parent

    subject_metrics: dict[int, dict[str, float]] = {}
    rows = []
    for subject, candidate_path in sorted(candidate_files.items()):
        print(f"Evaluating subject {subject}: {candidate_path}")
        with candidate_path.open("r", encoding="utf-8") as handle:
            candidates = json.load(handle)
        metrics = compute_caption_metrics(
            references,
            candidates,
            require_complete=strict,
            spice_corenlp_lib=spice_corenlp_lib,
        )
        subject_metrics[subject] = metrics
        subject_dir = output_dir / f"sub{subject:02d}"
        save_json(
            subject_dir / "metrics.json",
            {
                "subject": subject,
                "num_candidates": len(candidates),
                "candidate_path": str(candidate_path.resolve()),
                "reference_path": str(references_path.resolve()),
                "metrics": metrics,
            },
        )
        _write_metrics_text(subject_dir / "metrics.txt", metrics, candidate_path=candidate_path)
        rows.append({"subject": f"sub{subject:02d}", **metrics})

    average = aggregate_subject_metrics(subject_metrics)
    rows.append({"subject": "mean", **average})
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        output_dir / "summary.json",
        {
            "subjects": {f"sub{subject:02d}": metrics for subject, metrics in sorted(subject_metrics.items())},
            "mean": average,
        },
    )
    print(f"Saved evaluation summary: {summary_path}")
    return summary_path
