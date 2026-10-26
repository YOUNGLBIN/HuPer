"""Dependency-light smoke checks for environments where pytest is unavailable."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from huper.config import load_config
from huper.core.paths import PROJECT_DIR
from huper.evaluate import METRIC_NAMES, aggregate_subject_metrics, validate_caption_keys
from huper.formal.blur import DEFAULT_KERNELS, validate_gaussian_kernels
from huper.formal.fusion_model import MultiKernelBrainX
from huper.formal.gaussian_feature_cache import MultiKernelMQTFeatureCache, validate_multi_kernel_cache
from huper.train import fusion_kl_to_prior


class DummyBrain(nn.Module):
    def forward(self, value, modal="fmri1"):
        return value


def main() -> None:
    validate_gaussian_kernels(DEFAULT_KERNELS)
    config = load_config(PROJECT_DIR / "configs" / "paper.yaml")
    assert config["train"]["num_epochs"] == 160

    model = MultiKernelBrainX(
        brain=DummyBrain(),
        subjects=[1, 2],
        fusion_mode="token_wise",
        num_kernels=3,
        num_tokens=2,
        feat_dim=4,
        kernels=[1, 3, 9],
        init_std=0,
        fusion_prior_alpha=1,
    )
    features = torch.randn(1, 3, 2, 4)
    assert model.fuse_image_features(features, "fmri1").shape == (1, 2, 4)
    kl = fusion_kl_to_prior(model, modals=["fmri1", "fmri2"], kernels=[1, 3, 9], alpha=1)
    assert abs(kl.item()) < 1e-7

    with TemporaryDirectory() as temporary_dir:
        cache_path = Path(temporary_dir) / "cache.pt"
        metadata = {
            "feature_space": "mqt",
            "cache_type": "multi_kernel_gaussian_mqt",
            "use_image_aug": False,
            "mqt_num_tokens": 2,
            "kernels": [1, 3, 9],
            "complete": True,
        }
        cache_features = torch.randn(2, 3, 2, 4)
        torch.save(
            {
                "ids": torch.tensor([10, 20]),
                "features": cache_features,
                "metadata": metadata,
            },
            cache_path,
        )
        cache = MultiKernelMQTFeatureCache(cache_path, device=torch.device("cpu"))
        validate_multi_kernel_cache(Namespace(mqt_num_tokens=2, kernels=[1, 3, 9]), cache)
        assert torch.equal(cache.get([20], device=torch.device("cpu"))[0], cache_features[1])

    references = {"0": ["zero"], "1": ["one"]}
    assert validate_caption_keys(references, {"0": "zero"}, require_complete=False) == ["0"]
    metrics = {
        subject: {metric: float(subject) for metric in METRIC_NAMES}
        for subject in (1, 2, 5, 7)
    }
    average = aggregate_subject_metrics(metrics)
    assert all(value == 3.75 for value in average.values())
    print("manual smoke checks passed")


if __name__ == "__main__":
    main()
