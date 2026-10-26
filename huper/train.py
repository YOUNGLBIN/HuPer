#!/usr/bin/env python
"""Train BrainX with subject-specific token-wise multi-kernel Gaussian fusion."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from huper.core import training_utils
from huper.core.brain_encoder import BrainX
from huper.core.losses import token_alignment_loss
from huper.core.nsd_data import build_lr_scheduler, prepare_brainx_dataloaders
from huper.core.paths import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, save_json
from huper.formal.blur import DEFAULT_KERNELS, validate_gaussian_kernels
from huper.formal.fusion_model import MultiKernelBrainX
from huper.formal.gaussian_feature_cache import MultiKernelMQTFeatureCache, validate_multi_kernel_cache


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.backends.cuda.matmul.allow_tf32 = True

FUSION_MODE = "token_wise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="multi_kernel_token_wise_align")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--multi_kernel_feature_cache", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--subj", nargs="+", default=["1", "2", "5", "7"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--recon_loss", choices=["mse", "l1", "huber", "quantile"], default="mse")
    parser.add_argument("--use_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_umap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr_scheduler_type", choices=["cycle", "linear"], default="cycle")
    parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_at_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin_image_features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache_features_on_gpu", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--kernels", nargs="+", type=int, default=list(DEFAULT_KERNELS))
    parser.add_argument("--mqt_num_tokens", type=int, default=256)
    parser.add_argument("--fusion_init_std", type=float, default=0.01)
    parser.add_argument("--fusion_prior_alpha", type=float, default=1.0)
    parser.add_argument("--fusion_kl_weight", type=float, default=0.1)
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    args = parser.parse_args(argv)
    args.fusion_mode = FUSION_MODE
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.fusion_init_std < 0:
        raise ValueError("--fusion_init_std must be non-negative.")
    if args.fusion_prior_alpha < 0:
        raise ValueError("--fusion_prior_alpha must be non-negative.")
    if args.fusion_kl_weight < 0:
        raise ValueError("--fusion_kl_weight must be non-negative.")
    if not 1 <= args.mqt_num_tokens <= 256:
        raise ValueError("--mqt_num_tokens must be in [1, 256].")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch < 1:
        raise ValueError("--max_steps_per_epoch must be at least 1.")
    validate_gaussian_kernels(args.kernels)


def make_output_dir(args: argparse.Namespace, subs: list[int]) -> str:
    if args.model_save_path:
        return args.model_save_path

    date = datetime.datetime.now().strftime("%y%m%d")
    sub_tag = "sub" + "-".join(f"{sub:02d}" for sub in subs)
    run_name = "_".join(
        [
            args.model_name,
            sub_tag,
            FUSION_MODE,
            "gaussian",
            "cached",
            date,
        ]
    )
    return str(Path(args.output_root).expanduser() / run_name)


def optimizer_groups(model: MultiKernelBrainX):
    return training_utils.adamw_parameter_groups(
        model,
        no_decay_names=["bias", "LayerNorm.bias", "LayerNorm.weight", "fusion_logits"],
    )


def fusion_kl_to_prior(
    model: MultiKernelBrainX,
    *,
    modals: list[str],
    kernels: list[int],
    alpha: float,
) -> torch.Tensor:
    losses = []
    for modal in modals:
        logits = model.fusion_logits[modal]
        log_weights = torch.log_softmax(logits.float(), dim=0)
        weights = log_weights.exp()

        kernel_tensor = torch.tensor(kernels, device=logits.device, dtype=torch.float32)
        prior = kernel_tensor.pow(-float(alpha))
        prior = prior / prior.sum()
        log_prior = torch.log(prior).view(logits.shape[0], *([1] * (logits.ndim - 1)))

        kl_per_position = (weights * (log_weights - log_prior)).sum(dim=0)
        losses.append(kl_per_position.mean())

    if not losses:
        raise ValueError("modals must not be empty when computing fusion KL.")
    return torch.stack(losses).mean()


def collect_fusion_kernel_probs(model: MultiKernelBrainX, kernels: list[int]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for modal, logits in model.fusion_logits.items():
            values = torch.softmax(logits.detach().float(), dim=0)
            if values.ndim > 1:
                values = values.sum(dim=tuple(range(1, values.ndim)))
            values = values / values.sum().clamp_min(1e-12)
            payload[modal] = {
                f"k{kernel}": float(value)
                for kernel, value in zip(kernels, values.detach().cpu().tolist())
            }
    return payload


def format_fusion_kernel_probs(payload: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for modal, kernel_probs in payload.items():
        body = " ".join(f"{kernel}={value:.6f}" for kernel, value in kernel_probs.items())
        lines.append(f"{modal}: {body}")
    return lines


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    print_fn = accelerator.print
    print_fn("PID of this process =", os.getpid())
    device = accelerator.device
    print_fn("device:", device)
    print_fn(accelerator.state)

    num_devices = accelerator.num_processes
    num_workers = max(1, num_devices)
    local_rank = accelerator.state.local_process_index

    subs = [int(sub) for sub in args.subj]
    supported_subs = {1, 2, 5, 7}
    unsupported = sorted(set(subs) - supported_subs)
    if unsupported:
        raise ValueError(f"BrainX supports subjects {sorted(supported_subs)}, got unsupported subjects {unsupported}.")
    if len(subs) < 2:
        raise ValueError("Cross-subject training requires at least two subjects.")

    outdir = make_output_dir(args, subs)
    output_path = Path(outdir)
    if output_path.exists() and any(output_path.iterdir()) and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Training output directory is not empty: {output_path}. "
            "Use --resume with an existing checkpoint or choose a new directory."
        )
    os.makedirs(outdir, exist_ok=True)

    training_utils.seed_everything(args.seed, cudnn_deterministic=False)

    print_fn("\nloading multi-kernel MQT feature cache...")
    cache_device = device if args.cache_features_on_gpu else torch.device("cpu")
    mqt_feature_cache = MultiKernelMQTFeatureCache(
        args.multi_kernel_feature_cache,
        device=cache_device,
        pin_memory=args.pin_image_features,
    )
    validate_multi_kernel_cache(args, mqt_feature_cache)
    feat_dim = mqt_feature_cache.feat_dim
    num_tokens = mqt_feature_cache.num_tokens
    num_kernels = mqt_feature_cache.num_kernels
    target_metadata = mqt_feature_cache.metadata
    print_fn(
        f"target cache={args.multi_kernel_feature_cache}, kernels={num_kernels}, "
        f"tokens={num_tokens}, feat_dim={feat_dim}, cache_device={cache_device}, fusion={FUSION_MODE}"
    )

    print_fn("\nprepare NSD webdataset data...")
    print_fn(f"training with subs: {subs}")
    train_dls, val_dls, num_train = prepare_brainx_dataloaders(
        args,
        subs,
        args.batch_size,
        num_devices,
        num_workers,
        print_fn,
    )

    print_fn("\ncreating token-wise MultiKernelBrainX...")
    brain = BrainX(
        hidden_dim=1024,
        out_dim=feat_dim,
        num_latents=num_tokens,
        use_norm=args.use_norm,
        use_token=args.use_token,
    )
    voxel2emb = MultiKernelBrainX(
        brain=brain,
        subjects=subs,
        fusion_mode=FUSION_MODE,
        num_kernels=num_kernels,
        num_tokens=num_tokens,
        feat_dim=feat_dim,
        kernels=args.kernels,
        pixel_map_size=None,
        init_std=args.fusion_init_std,
        fusion_prior_alpha=args.fusion_prior_alpha,
    ).to(device)

    if local_rank == 0:
        training_utils.print_parameter_count(voxel2emb)
        print_fn("fusion summary:", voxel2emb.fusion_summary())

    if args.dry_run:
        (voxel, coco) = next(iter(train_dls[0]))
        voxel = voxel[:, 0].to(device, non_blocking=True).float()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prediction = voxel2emb(voxel, modal=f"fmri{subs[0]}")
            target_features = mqt_feature_cache.get(coco, device=device)
            target = voxel2emb.fuse_image_features(target_features, modal=f"fmri{subs[0]}")
        if prediction.shape != target.shape:
            raise RuntimeError(f"Dry-run shape mismatch: {prediction.shape} vs {target.shape}")
        print_fn(f"dry-run successful: prediction/target shape={tuple(prediction.shape)}")
        return

    voxel2emb.requires_grad_(True)
    optimizer = torch.optim.AdamW(optimizer_groups(voxel2emb), lr=args.max_lr)
    lr_scheduler = build_lr_scheduler(args, optimizer, num_train, args.batch_size, num_devices, args.num_epochs)

    args_dict = vars(args).copy()
    args_dict.update(
        {
            "subs": subs,
            "feat_dim": feat_dim,
            "num_tokens": num_tokens,
            "num_kernels": num_kernels,
            "pixel_map_size": None,
            "outdir": outdir,
            "target_metadata": target_metadata,
            "fusion_summary": voxel2emb.fusion_summary(),
            "align_mode": "raw",
            "feature_space": "mqt",
            "use_image_aug": False,
        }
    )
    save_json(Path(outdir) / "config.json", args_dict)

    epoch = 0
    losses, val_losses, lrs = [], [], []
    best_ckpt_metric = "val/loss" if args.eval_val else "train/loss"
    best_ckpt_loss = 1e9

    if args.resume:
        print_fn(f"loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "formal_model_state_dict" in checkpoint:
            voxel2emb.load_state_dict(checkpoint["formal_model_state_dict"])
        elif "model_state_dict" in checkpoint:
            voxel2emb.brain.load_state_dict(checkpoint["model_state_dict"])
            print_fn("loaded BrainX weights from legacy checkpoint; fusion logits keep their new initialization")
        else:
            raise KeyError(f"Checkpoint {args.resume} does not contain model weights.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch = checkpoint["epoch"] + 1
        losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        lrs = checkpoint.get("lrs", [])
        saved_best_metric = checkpoint.get("best_ckpt_metric")
        if saved_best_metric == best_ckpt_metric:
            best_ckpt_loss = checkpoint.get("best_ckpt_loss", checkpoint.get("best_val_loss", 1e9))
        elif args.eval_val:
            best_ckpt_loss = min(val_losses) if val_losses else 1e9
        else:
            best_ckpt_loss = checkpoint.get("best_train_loss", 1e9)

    voxel2emb, optimizer, lr_scheduler = accelerator.prepare(voxel2emb, optimizer, lr_scheduler)
    loss_fn = training_utils.build_loss_fn(args.recon_loss)

    def make_image_target(coco_ids, modal: str) -> torch.Tensor:
        kernel_features = mqt_feature_cache.get(coco_ids, device=device)
        return voxel2emb.fuse_image_features(kernel_features, modal=modal)

    def save_ckpt(tag):
        ckpt_path = os.path.join(outdir, f"{tag}.pth")
        print_fn(f"\nsaving {ckpt_path}", flush=True)
        unwrapped_model = accelerator.unwrap_model(voxel2emb)
        temporary_path = f"{ckpt_path}.tmp"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": unwrapped_model.brain.state_dict(),
                "formal_model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_losses": losses,
                "val_losses": val_losses,
                "best_ckpt_metric": best_ckpt_metric,
                "best_ckpt_loss": best_ckpt_loss,
                "best_val_loss": best_ckpt_loss if args.eval_val else None,
                "best_train_loss": best_ckpt_loss if not args.eval_val else None,
                "lrs": lrs,
                "config": args_dict,
            },
            temporary_path,
        )
        os.replace(temporary_path, ckpt_path)
        del unwrapped_model

    print_fn(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
    progress_bar = tqdm(range(epoch, args.num_epochs), ncols=120, disable=(local_rank != 0))

    for epoch in progress_bar:
        voxel2emb.train()
        loss_full_sum = 0.0
        loss_recon_sum = 0.0
        loss_kl_sum = 0.0
        loss_kl_weighted_sum = 0.0

        selected_sub = random.sample(subs, 2)
        selected_idx = [subs.index(item) for item in selected_sub]
        train_dl_0 = train_dls[selected_idx[0]]
        train_dl_1 = train_dls[selected_idx[1]]
        selected_modals = [f"fmri{sub}" for sub in selected_sub]

        train_i = -1
        for train_i, ((voxel_0, coco_0), (voxel_1, coco_1)) in enumerate(zip(train_dl_0, train_dl_1)):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                optimizer.zero_grad()

                repeat_index = train_i % 3
                voxel_0 = voxel_0[:, repeat_index].to(device, non_blocking=True).float()
                voxel_1 = voxel_1[:, repeat_index].to(device, non_blocking=True).float()

                emb_voxel_0 = voxel2emb(voxel_0, modal=f"fmri{selected_sub[0]}")
                emb_voxel_1 = voxel2emb(voxel_1, modal=f"fmri{selected_sub[1]}")
                emb_image_0 = make_image_target(coco_0, modal=f"fmri{selected_sub[0]}")
                emb_image_1 = make_image_target(coco_1, modal=f"fmri{selected_sub[1]}")

                loss_0 = token_alignment_loss(emb_voxel_0, emb_image_0, loss_fn=loss_fn)
                loss_1 = token_alignment_loss(emb_voxel_1, emb_image_1, loss_fn=loss_fn)
                loss_mse = loss_0 + loss_1
                loss_kl = fusion_kl_to_prior(
                    accelerator.unwrap_model(voxel2emb),
                    modals=selected_modals,
                    kernels=args.kernels,
                    alpha=args.fusion_prior_alpha,
                )
                loss_kl_weighted = args.fusion_kl_weight * loss_kl
                loss = loss_mse + loss_kl_weighted
                training_utils.check_finite_loss(loss)

            loss_full_sum += loss.item()
            loss_recon_sum += loss_mse.item()
            loss_kl_sum += loss_kl.item()
            loss_kl_weighted_sum += loss_kl_weighted.item()
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])
            if args.lr_scheduler_type is not None:
                lr_scheduler.step()
            if args.max_steps_per_epoch is not None and train_i + 1 >= args.max_steps_per_epoch:
                break

        if train_i < 0:
            raise RuntimeError("No training batches were produced.")

        train_epoch_loss = loss_full_sum / (train_i + 1)
        train_epoch_mse_loss = loss_recon_sum / (train_i + 1)
        train_epoch_kl_loss = loss_kl_sum / (train_i + 1)
        train_epoch_weighted_kl_loss = loss_kl_weighted_sum / (train_i + 1)

        val_loss = None
        val_logs_by_sub = {}
        if args.eval_val:
            voxel2emb.eval()
            val_subject_losses = []
            for val_sub, val_dl in val_dls:
                val_loss_recon_sum = 0.0
                val_sub_steps = 0

                for voxel, coco in val_dl:
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        voxel = torch.mean(voxel, axis=1).to(device, non_blocking=True).float()
                        emb_voxel = voxel2emb(voxel, modal=f"fmri{val_sub}")
                        emb_image = make_image_target(coco, modal=f"fmri{val_sub}")
                        val_loss_recon = token_alignment_loss(
                            emb_voxel,
                            emb_image,
                            loss_fn=loss_fn,
                        )
                        training_utils.check_finite_loss(val_loss_recon)

                    val_loss_recon_sum += val_loss_recon.item()
                    val_losses.append(val_loss_recon.item())
                    val_sub_steps += 1

                if val_sub_steps > 0:
                    val_sub_loss = val_loss_recon_sum / val_sub_steps
                    val_subject_losses.append(val_sub_loss)
                    val_logs_by_sub[f"val/loss_subj{val_sub:02d}"] = val_sub_loss

            val_loss = float(np.mean(val_subject_losses))

        if local_rank == 0:
            if (not args.save_at_end and args.ckpt_saving) or (args.save_at_end and epoch == args.num_epochs - 1):
                best_candidate_loss = val_loss if args.eval_val else train_epoch_loss
                if best_candidate_loss < best_ckpt_loss:
                    best_ckpt_loss = best_candidate_loss
                    save_ckpt("best")
                else:
                    print_fn(
                        f"not best - {best_ckpt_metric}: {best_candidate_loss:.3f}, "
                        f"best_{best_ckpt_metric}: {best_ckpt_loss:.3f}"
                    )

            logs = {
                "train/loss": train_epoch_loss,
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "train/selected_sub_0": selected_sub[0],
                "train/selected_sub_1": selected_sub[1],
                "train/loss_mse": train_epoch_mse_loss,
                "train/loss_recon": train_epoch_mse_loss,
                "train/loss_kl": train_epoch_kl_loss,
                "train/loss_kl_weighted": train_epoch_weighted_kl_loss,
                "val/eval_enabled": args.eval_val,
                "best/metric": best_ckpt_metric,
                "best/loss": best_ckpt_loss,
                "fusion/mode": FUSION_MODE,
                "fusion/kl_weight": args.fusion_kl_weight,
            }
            if args.eval_val:
                logs.update(
                    {
                        "val/loss": val_loss,
                        "val/loss_mse": val_loss,
                        "val/loss_recon": val_loss,
                        "val/num_steps": len(val_losses),
                    }
                )
            logs.update(val_logs_by_sub)
            progress_bar.set_postfix(**logs)

            fusion_kernel_probs = collect_fusion_kernel_probs(
                accelerator.unwrap_model(voxel2emb),
                args.kernels,
            )

            with open(os.path.join(outdir, "logs.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, **logs}) + "\n")

            with open(os.path.join(outdir, "fusion_weight_sums.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch}\n")
                for line in format_fusion_kernel_probs(fusion_kernel_probs):
                    f.write(f"  {line}\n")

            save_ckpt("last")

        accelerator.wait_for_everyone()

    if local_rank == 0:
        training_utils.plot_curve(losses, "train loss", Path(outdir) / "loss_train.png")
        if args.eval_val and val_losses:
            training_utils.plot_curve(val_losses, "val loss", Path(outdir) / "loss_val.png")
        training_utils.plot_curve(lrs, "lr", Path(outdir) / "lr.png")


if __name__ == "__main__":
    main()
