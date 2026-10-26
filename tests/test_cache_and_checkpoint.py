import json
from argparse import Namespace

import pytest
import torch

from huper.core.mqt_captioning import bool_from_config, read_checkpoint_config
from huper.formal.gaussian_feature_cache import (
    MultiKernelMQTFeatureCache,
    validate_multi_kernel_cache,
)


def test_cache_lookup_and_metadata_validation(tmp_path):
    cache_path = tmp_path / "cache.pt"
    metadata = {
        "feature_space": "mqt",
        "cache_type": "multi_kernel_gaussian_mqt",
        "use_image_aug": False,
        "mqt_num_tokens": 2,
        "kernels": [1, 3, 9],
        "complete": True,
    }
    features = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).half()
    torch.save(
        {
            "ids": torch.tensor([10, 20]),
            "features": features,
            "metadata": metadata,
        },
        cache_path,
    )

    cache = MultiKernelMQTFeatureCache(cache_path, device=torch.device("cpu"))
    args = Namespace(mqt_num_tokens=2, kernels=[1, 3, 9])
    validate_multi_kernel_cache(args, cache)
    selected = cache.get(torch.tensor([20, 10]), device=torch.device("cpu"))
    assert torch.equal(selected[0], features[1])
    assert torch.equal(selected[1], features[0])


def test_partial_cache_is_rejected(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "ids": torch.tensor([10]),
            "features": torch.zeros(1, 1, 1, 1),
            "metadata": {
                "feature_space": "mqt",
                "cache_type": "multi_kernel_gaussian_mqt",
                "use_image_aug": False,
                "mqt_num_tokens": 1,
                "kernels": [1],
                "complete": False,
            },
        },
        cache_path,
    )
    cache = MultiKernelMQTFeatureCache(cache_path, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="partial"):
        validate_multi_kernel_cache(Namespace(mqt_num_tokens=1, kernels=[1]), cache)


def test_checkpoint_config_round_trip(tmp_path):
    checkpoint_path = tmp_path / "last.pth"
    checkpoint_path.touch()
    config = {"use_norm": False, "use_token": True}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    config_path, loaded = read_checkpoint_config(checkpoint_path)
    assert config_path == tmp_path / "config.json"
    assert bool_from_config(loaded, "use_norm", config_path) is False
    assert bool_from_config(loaded, "use_token", config_path) is True
