#!/usr/bin/env python
"""Caption inference for formal multi-kernel blur alignment checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from huper.core import mqt_captioning as mqt_infer
from huper.core import training_utils
from huper.core.brain_encoder import BrainX
from huper.core.nsd_data import make_caption_val_loader
from huper.core.paths import DEFAULT_DATA_PATH, DEFAULT_MQT_MODEL_PATH, resolve_hf_model_path, save_json
from huper.formal.fusion_model import MultiKernelBrainX


DEFAULT_PROMPT = mqt_infer.DEFAULT_PROMPT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainx_path", required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--save_path", default="results")
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--mqt_model_path", default=DEFAULT_MQT_MODEL_PATH)
    parser.add_argument("--mqt_token_stage", choices=["pre_proj", "post_proj"], default="pre_proj")
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_name", default="MQT-LLaVA-7b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _config_value(config: dict, summary: dict, key: str, default=None):
    if key in config:
        return config[key]
    return summary.get(key, default)


def apply_checkpoint_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("Inference batch sizes must be at least 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1.")

    config_path, config = mqt_infer.read_checkpoint_config(args.brainx_path)
    summary = config.get("fusion_summary", {})

    args.training_config_path = str(config_path)
    args.use_norm = mqt_infer.bool_from_config(config, "use_norm", config_path)
    args.use_token = mqt_infer.bool_from_config(config, "use_token", config_path)
    args.feat_dim = int(_config_value(config, summary, "feat_dim", 1024))
    args.num_tokens = int(_config_value(config, summary, "num_tokens", 256))
    args.fusion_mode = str(_config_value(config, summary, "fusion_mode", "token_wise"))

    kernels = _config_value(config, summary, "kernels", None)
    if kernels is None:
        num_kernels = int(_config_value(config, summary, "num_kernels", 12))
        kernels = list(range(1, num_kernels + 1))
    args.kernels = [int(kernel) for kernel in kernels]
    args.num_kernels = int(_config_value(config, summary, "num_kernels", len(args.kernels)))

    subjects = _config_value(config, summary, "subs", None)
    if subjects is None:
        subjects = _config_value(config, summary, "subjects", [1, 2, 5, 7])
    args.subjects = [int(sub) for sub in subjects]

    pixel_map_size = _config_value(config, summary, "pixel_map_size", None)
    args.pixel_map_size = tuple(int(value) for value in pixel_map_size) if pixel_map_size else None
    args.fusion_init_std = float(_config_value(config, summary, "fusion_init_std", 0.0))
    args.fusion_prior_alpha = float(_config_value(config, summary, "fusion_prior_alpha", 0.5))

    if args.subj not in args.subjects:
        raise ValueError(f"Checkpoint was trained for subjects {args.subjects}, but --subj={args.subj}.")
    return args


def build_formal_model(args: argparse.Namespace) -> MultiKernelBrainX:
    brain = BrainX(
        hidden_dim=1024,
        out_dim=args.feat_dim,
        num_latents=args.num_tokens,
        use_norm=args.use_norm,
        use_token=args.use_token,
    )
    return MultiKernelBrainX(
        brain=brain,
        subjects=args.subjects,
        fusion_mode=args.fusion_mode,
        num_kernels=args.num_kernels,
        num_tokens=args.num_tokens,
        feat_dim=args.feat_dim,
        kernels=args.kernels,
        pixel_map_size=args.pixel_map_size,
        init_std=args.fusion_init_std,
        fusion_prior_alpha=args.fusion_prior_alpha,
    )


def load_formal_checkpoint(model: MultiKernelBrainX, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "formal_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["formal_model_state_dict"])
        return

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.brain.load_state_dict(state_dict)


def predict_mqt_tokens(voxel2emb, val_loader, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    token_batches = []
    seen = 0

    for voxel, _ in val_loader:
        if args.max_samples is not None:
            remaining = args.max_samples - seen
            if remaining <= 0:
                break
            voxel = voxel[:remaining]

        voxel = torch.mean(voxel, dim=1).float().to(device)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            tokens = voxel2emb(voxel, modal=f"fmri{args.subj}")

        token_batches.append(tokens.detach().cpu())
        seen += tokens.shape[0]

    if not token_batches:
        raise RuntimeError("No samples were loaded from the validation dataset.")
    return torch.cat(token_batches, dim=0)


def caption_formal_batch(tokens, prompt_input_ids, tokenizer, model, args: argparse.Namespace):
    with torch.inference_mode():
        visual_tokens = mqt_infer.ensure_post_proj_tokens(tokens, model, args.mqt_token_stage)
        inputs_embeds = mqt_infer.build_inputs_embeds(prompt_input_ids, visual_tokens, model)
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "do_sample": args.temperature > 0,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                generation_kwargs["top_p"] = args.top_p

        output_ids = mqt_infer.generate_with_inputs_embeds(model, **generation_kwargs)
    return [text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def _prepare_caption_outputs(result_dir: Path, args) -> tuple[Path, Path, dict[str, str]]:
    response_path = result_dir / "response.txt"
    caption_path = result_dir / "fmricap.json"
    existing = response_path.exists() or caption_path.exists()

    if existing and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Inference output already exists under {result_dir}. "
            "Pass --resume to continue or --overwrite to replace it."
        )

    if args.overwrite or not existing:
        response_path.write_text("", encoding="utf-8")
        return response_path, caption_path, {}

    if caption_path.exists():
        with caption_path.open("r", encoding="utf-8") as handle:
            captions = json.load(handle)
    else:
        captions = {}
    expected_keys = [str(index) for index in range(len(captions))]
    if list(captions) != expected_keys:
        raise ValueError("Resume requires fmricap.json to contain a contiguous prefix starting at key '0'.")
    response_path.touch(exist_ok=True)
    return response_path, caption_path, captions


def main(argv: list[str] | None = None) -> None:
    args = apply_checkpoint_config(parse_args(argv))
    args.mqt_model_path = str(resolve_hf_model_path(args.mqt_model_path))
    training_utils.seed_everything(args.seed, cudnn_deterministic=False)

    result_dir = Path(args.save_path).expanduser() / f"sub{args.subj:02d}_dim{args.feat_dim}"
    result_dir.mkdir(parents=True, exist_ok=True)
    response_path, caption_path, cap_result = _prepare_caption_outputs(result_dir, args)

    print("prepare NSD webdataset data...")
    val_loader = make_caption_val_loader(args.data_path, args.subj, args.val_batch_size)

    print("load formal BrainX checkpoint...")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    fmri_device = requested_device
    voxel2emb = build_formal_model(args).to(fmri_device)
    load_formal_checkpoint(voxel2emb, args.brainx_path)
    voxel2emb.eval()

    print("predict MQT pre-projection tokens from fMRI...")
    target_tokens = predict_mqt_tokens(voxel2emb, val_loader, args, fmri_device)
    print(f"target_tokens.shape: {tuple(target_tokens.shape)}")

    del voxel2emb, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("load MQT-LLaVA...")
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.utils import disable_torch_init

    disable_torch_init()
    tokenizer, llava_model, _, _ = mqt_infer.load_mqt_llava_model(args)
    prompt = mqt_infer.build_prompt(args.prompt, llava_model.config.mm_use_im_start_end)
    prompt_input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    with torch.inference_mode():
        preview_end = min(int(args.batch_size), int(target_tokens.shape[0]))
        preview_visual_tokens = mqt_infer.ensure_post_proj_tokens(
            target_tokens[:preview_end],
            llava_model,
            args.mqt_token_stage,
        )
    print(f"visual_tokens.shape after projection: {tuple(preview_visual_tokens.shape)}")
    del preview_visual_tokens

    save_json(
        result_dir / "inference_config.json",
        {
            **vars(args),
            "target_feature_shape": list(target_tokens.shape),
            "visual_token_dim": int(llava_model.config.hidden_size),
        },
    )

    total = int(target_tokens.shape[0])
    if len(cap_result) > total:
        raise ValueError(f"Existing caption count {len(cap_result)} exceeds requested total {total}.")
    for start in range(len(cap_result), total, args.batch_size):
        end = min(start + args.batch_size, total)
        start_time = time.time()
        captions = caption_formal_batch(
            target_tokens[start:end],
            prompt_input_ids,
            tokenizer,
            llava_model,
            args,
        )
        print(f"generated [{start}:{end}] in {time.time() - start_time:.2f} seconds")

        with response_path.open("a", encoding="utf-8") as f:
            for offset, caption in enumerate(captions):
                idx = start + offset
                f.write(f"response_{idx}: \n")
                f.write(caption + "\n")
                cap_result[str(idx)] = caption
                print(f"[{idx + 1}/{total}] {caption}")

        if args.save_every > 0 and (end % args.save_every == 0 or end == total):
            save_json(caption_path, cap_result)

    save_json(caption_path, cap_result)
    inference_config = {
        **vars(args),
        "target_feature_shape": list(target_tokens.shape),
        "visual_token_dim": int(llava_model.config.hidden_size),
        "num_samples": len(cap_result),
    }
    save_json(result_dir / "inference_config.json", inference_config)
    print(f"saved {len(cap_result)} captions: {caption_path}")


if __name__ == "__main__":
    main()
