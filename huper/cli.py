"""Unified command-line interface for the HuPer reproduction pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huper import __version__
from huper.config import (
    load_config,
    prepare_runtime_environment,
    require_nsd_root,
    resolve_mqt_snapshot,
)
from huper.core.paths import PROJECT_DIR


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "paper.yaml"
STAGES = ("preprocess", "train", "infer", "evaluate")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the paper YAML configuration.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="huper", description=__doc__)
    parser.add_argument("--version", action="version", version=f"HuPer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-models",
        help="Download pinned MQT-LLaVA, CLIP-336, and SPICE CoreNLP assets.",
    )
    _add_config(download)

    check = subparsers.add_parser("check", help="Validate environment, NSD shards, models, Java, and disk space.")
    _add_config(check)

    preprocess = subparsers.add_parser("preprocess", help="Build the Gaussian MQT feature cache.")
    _add_config(preprocess)
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument("--max-images", type=int, default=None, help="Smoke-test limit; creates an incomplete cache.")

    train = subparsers.add_parser("train", help="Train formal token-wise HuPer.")
    _add_config(train)
    train.add_argument("--resume", default=None, help="Checkpoint to resume.")
    train.add_argument("--dry-run", action="store_true", help="Run one forward/cache validation without optimization.")
    train.add_argument("--max-steps-per-epoch", type=int, default=None, help="Debug-only optimizer-step limit.")
    train.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; defaults to physical GPU 0.")

    infer = subparsers.add_parser("infer", help="Generate captions for one or more paper subjects.")
    _add_config(infer)
    infer.add_argument("--checkpoint", default=None)
    infer.add_argument("--subjects", nargs="+", type=int, default=None)
    infer.add_argument("--max-samples", type=int, default=None)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; defaults to physical GPU 0.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate fmricap.json files with BrainHub metrics.")
    _add_config(evaluate)
    evaluate.add_argument("--input", default=None, help="Caption file or root containing one file per subject.")
    evaluate.add_argument("--allow-partial", action="store_true", help="Permit partial captions for smoke tests.")

    run_all = subparsers.add_parser("run-all", help="Run preprocess, train, infer, and evaluate in order.")
    _add_config(run_all)
    run_all.add_argument("--from-stage", choices=STAGES, default="preprocess")
    run_all.add_argument("--overwrite-cache", action="store_true")
    run_all.add_argument("--resume-inference", action="store_true")
    run_all.add_argument("--overwrite-inference", action="store_true")
    run_all.add_argument("--max-samples", type=int, default=None)
    run_all.add_argument("--gpu", default="0")
    return parser


def _set_gpu(gpu: str) -> None:
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def _bool_option(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _train_argv(config: dict, args) -> list[str]:
    experiment = config["experiment"]
    training = config["train"]
    argv = [
        "--model_name",
        str(training["run_name"]),
        "--data_path",
        str(require_nsd_root(config)),
        "--multi_kernel_feature_cache",
        str(training["feature_cache"]),
        "--model_save_path",
        str(training["output_dir"]),
        "--subj",
        *[str(subject) for subject in experiment["subjects"]],
        "--batch_size",
        str(training["batch_size"]),
        "--val_batch_size",
        str(training["val_batch_size"]),
        "--num_epochs",
        str(training["num_epochs"]),
        "--seed",
        str(experiment["seed"]),
        "--max_lr",
        str(training["max_lr"]),
        "--recon_loss",
        str(training["recon_loss"]),
        _bool_option("use_norm", bool(training["use_norm"])),
        _bool_option("use_token", bool(training["use_token"])),
        _bool_option("eval_val", bool(training["eval_val"])),
        "--kernels",
        *[str(kernel) for kernel in experiment["kernels"]],
        "--mqt_num_tokens",
        str(experiment["mqt_num_tokens"]),
        "--fusion_init_std",
        str(training["fusion_init_std"]),
        "--fusion_prior_alpha",
        str(training["fusion_prior_alpha"]),
        "--fusion_kl_weight",
        str(training["fusion_kl_weight"]),
        _bool_option("cache_features_on_gpu", bool(training["cache_features_on_gpu"])),
    ]
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = PROJECT_DIR / resume_path
        argv.extend(["--resume", str(resume_path.resolve())])
    if args.dry_run:
        argv.append("--dry_run")
    if args.max_steps_per_epoch is not None:
        argv.extend(["--max_steps_per_epoch", str(args.max_steps_per_epoch)])
    return argv


def _run_train(config: dict, args) -> None:
    _set_gpu(args.gpu)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["paths"]["output_root"]) / ".matplotlib"))
    from huper.train import main as train_main

    train_main(_train_argv(config, args))


def _inference_argv(config: dict, args, subject: int) -> list[str]:
    experiment = config["experiment"]
    inference = config["inference"]
    checkpoint = args.checkpoint or inference["checkpoint"]
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path

    argv = [
        "--brainx_path",
        str(checkpoint_path.resolve()),
        "--data_path",
        str(require_nsd_root(config)),
        "--save_path",
        str(inference["output_dir"]),
        "--subj",
        str(subject),
        "--seed",
        str(experiment["seed"]),
        "--val_batch_size",
        str(inference["val_batch_size"]),
        "--mqt_model_path",
        str(resolve_mqt_snapshot(config)),
        "--mqt_token_stage",
        "pre_proj",
        "--model_name",
        "MQT-LLaVA-7b",
        "--prompt",
        str(inference["prompt"]),
        "--batch_size",
        str(inference["batch_size"]),
        "--temperature",
        str(inference["temperature"]),
        "--num_beams",
        str(inference["num_beams"]),
        "--max_new_tokens",
        str(inference["max_new_tokens"]),
        "--device",
        str(inference["device"]),
    ]
    if inference.get("top_p") is not None:
        argv.extend(["--top_p", str(inference["top_p"])])
    if args.max_samples is not None:
        argv.extend(["--max_samples", str(args.max_samples)])
    if args.resume:
        argv.append("--resume")
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def _run_infer(config: dict, args) -> None:
    _set_gpu(args.gpu)
    subjects = args.subjects or [int(subject) for subject in config["experiment"]["subjects"]]
    unsupported = sorted(set(subjects) - {1, 2, 5, 7})
    if unsupported:
        raise ValueError(f"Unsupported subject(s): {unsupported}")
    from huper.infer import main as infer_main

    for subject in subjects:
        print(f"\n=== HuPer inference: subject {subject} ===")
        infer_main(_inference_argv(config, args, subject))


def _run_evaluate(config: dict, args) -> None:
    from huper.evaluate import run_evaluate

    run_evaluate(
        config,
        input_path=args.input,
        require_complete=not args.allow_partial,
    )


def _run_all(config: dict, args) -> None:
    start_index = STAGES.index(args.from_stage)
    stages = STAGES[start_index:]
    if "preprocess" in stages:
        from huper.preprocess import run_preprocess

        run_preprocess(config, overwrite=args.overwrite_cache)
    if "train" in stages:
        train_args = argparse.Namespace(
            resume=None,
            dry_run=False,
            max_steps_per_epoch=None,
            gpu=args.gpu,
        )
        _run_train(config, train_args)
    if "infer" in stages:
        infer_args = argparse.Namespace(
            checkpoint=None,
            subjects=None,
            max_samples=args.max_samples,
            resume=args.resume_inference,
            overwrite=args.overwrite_inference,
            gpu=args.gpu,
        )
        _run_infer(config, infer_args)
    if "evaluate" in stages:
        evaluate_args = argparse.Namespace(
            input=None,
            allow_partial=args.max_samples is not None,
        )
        _run_evaluate(config, evaluate_args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "download-models":
        from huper.assets import download_models

        download_models(config)
        return

    prepare_runtime_environment(config)
    if args.command == "check":
        from huper.checks import run_checks

        run_checks(config)
    elif args.command == "preprocess":
        from huper.preprocess import run_preprocess

        run_preprocess(config, overwrite=args.overwrite, max_images=args.max_images)
    elif args.command == "train":
        _run_train(config, args)
    elif args.command == "infer":
        _run_infer(config, args)
    elif args.command == "evaluate":
        _run_evaluate(config, args)
    elif args.command == "run-all":
        _run_all(config, args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
