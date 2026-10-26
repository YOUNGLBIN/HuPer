#!/usr/bin/env python3
"""MQT-LLaVA captioning helpers used by HuPer inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from huper.config import PAPER_PROMPT
from huper.core.paths import DEFAULT_MQT_MODEL_PATH


VOXELS_PER_SUBJ = {
    1: 15724,
    2: 14278,
    3: 15226,
    4: 13153,
    5: 13039,
    6: 17907,
    7: 12682,
    8: 14386,
}

DEFAULT_PROMPT = PAPER_PROMPT


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_checkpoint_config(checkpoint_path):
    config_path = Path(checkpoint_path).expanduser().parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected training config next to checkpoint: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return config_path, json.load(f)


def bool_from_config(config, key, config_path):
    if key not in config:
        raise KeyError(f"Missing {key!r} in {config_path}")

    value = config[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str2bool(value)
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Invalid boolean value for {key!r} in {config_path}: {value!r}")




def patch_hf_loaders():
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

    originals = {
        "tokenizer": AutoTokenizer.from_pretrained,
        "processor": CLIPImageProcessor.from_pretrained,
        "clip": CLIPVisionModel.from_pretrained,
    }

    def tokenizer_from_pretrained(*tokenizer_args, **tokenizer_kwargs):
        if tokenizer_kwargs.get("use_fast") is False:
            tokenizer_kwargs.setdefault("legacy", True)
        tokenizer_kwargs.setdefault("local_files_only", True)
        return originals["tokenizer"](*tokenizer_args, **tokenizer_kwargs)

    def local_only(original_func):
        def wrapper(*from_pretrained_args, **from_pretrained_kwargs):
            from_pretrained_kwargs.setdefault("local_files_only", True)
            return original_func(*from_pretrained_args, **from_pretrained_kwargs)

        return wrapper

    AutoTokenizer.from_pretrained = tokenizer_from_pretrained
    CLIPImageProcessor.from_pretrained = local_only(originals["processor"])
    CLIPVisionModel.from_pretrained = local_only(originals["clip"])
    return AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, originals


def restore_hf_loaders(AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, originals):
    AutoTokenizer.from_pretrained = originals["tokenizer"]
    CLIPImageProcessor.from_pretrained = originals["processor"]
    CLIPVisionModel.from_pretrained = originals["clip"]


def load_mqt_llava_model(args):
    from llava.model.builder import load_pretrained_model

    AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, originals = patch_hf_loaders()
    try:
        device = args.device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.")
            device = "cpu"

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.mqt_model_path or DEFAULT_MQT_MODEL_PATH,
            model_base=args.model_base,
            model_name=args.model_name,
            device=device,
            local_files_only=True,
        )
        if device == "cpu":
            model.float()
        return tokenizer, model.eval(), image_processor, context_len
    finally:
        restore_hf_loaders(AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, originals)


def build_prompt(query, mm_use_im_start_end):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER
    from llava.conversation import conv_templates

    if mm_use_im_start_end:
        raise ValueError("This helper expects MQT-LLaVA-7b without image start/end tokens.")

    if IMAGE_PLACEHOLDER in query:
        query = query.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
    elif DEFAULT_IMAGE_TOKEN not in query:
        query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def model_input_dtype(model):
    return model.get_model().embed_tokens.weight.dtype


def ensure_post_proj_tokens(tokens, model, mqt_token_stage):
    tokens = tokens.to(device=model.device, dtype=model_input_dtype(model))
    hidden_size = int(model.config.hidden_size)

    if mqt_token_stage == "post_proj":
        if tokens.shape[-1] != hidden_size:
            raise ValueError(f"post_proj tokens must have dim {hidden_size}, got {tokens.shape[-1]}.")
        return tokens

    if mqt_token_stage == "pre_proj":
        proj = model.get_model().query_abstractor.proj.to(device=model.device, dtype=tokens.dtype)
        if tokens.shape[-1] != proj.shape[0]:
            raise ValueError(f"pre_proj tokens must have dim {proj.shape[0]}, got {tokens.shape[-1]}.")
        return tokens @ proj

    raise ValueError(f"Unsupported mqt_token_stage={mqt_token_stage!r}.")


def build_inputs_embeds(prompt_input_ids, visual_tokens, model):
    from llava.constants import IMAGE_TOKEN_INDEX

    image_positions = torch.where(prompt_input_ids == IMAGE_TOKEN_INDEX)[0]
    if image_positions.numel() != 1:
        raise ValueError(f"Prompt must contain exactly one <image> token, got {image_positions.numel()}.")

    image_pos = int(image_positions.item())
    text_before = prompt_input_ids[:image_pos].to(model.device)
    text_after = prompt_input_ids[image_pos + 1 :].to(model.device)
    before_embeds = model.get_model().embed_tokens(text_before)
    after_embeds = model.get_model().embed_tokens(text_after)

    return torch.stack(
        [torch.cat([before_embeds, cur_tokens, after_embeds], dim=0) for cur_tokens in visual_tokens],
        dim=0,
    )


def generate_with_inputs_embeds(model, **kwargs):
    return super(type(model), model).generate(**kwargs)
