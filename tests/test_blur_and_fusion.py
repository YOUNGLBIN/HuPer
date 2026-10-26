import pytest
import torch
from torch import nn

from huper.formal.blur import DEFAULT_KERNELS, validate_gaussian_kernels
from huper.formal.fusion_model import MultiKernelBrainX
from huper.train import fusion_kl_to_prior


class DummyBrain(nn.Module):
    def forward(self, value, modal="fmri1"):
        return value


def make_model():
    return MultiKernelBrainX(
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


def test_paper_kernels_are_valid():
    validate_gaussian_kernels(DEFAULT_KERNELS)
    with pytest.raises(ValueError):
        validate_gaussian_kernels([1, 2])


def test_token_wise_fusion_uses_kernel_prior():
    model = make_model()
    features = torch.stack(
        [torch.full((2, 4), float(value)) for value in (1, 3, 9)],
        dim=0,
    ).unsqueeze(0)
    fused = model.fuse_image_features(features, modal="fmri1")
    prior = torch.tensor([1.0, 1 / 3, 1 / 9])
    prior = prior / prior.sum()
    expected = sum(weight * value for weight, value in zip(prior, (1, 3, 9)))
    assert fused.shape == (1, 2, 4)
    assert torch.allclose(fused, torch.full_like(fused, expected))


def test_kl_is_zero_at_exact_prior():
    model = make_model()
    loss = fusion_kl_to_prior(
        model,
        modals=["fmri1", "fmri2"],
        kernels=[1, 3, 9],
        alpha=1,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_archived_fusion_modes_are_rejected():
    with pytest.raises(ValueError, match="token_wise"):
        MultiKernelBrainX(
            brain=DummyBrain(),
            subjects=[1],
            fusion_mode="feature_dim_wise",
            num_kernels=3,
            num_tokens=2,
            feat_dim=4,
            kernels=[1, 3, 9],
        )
