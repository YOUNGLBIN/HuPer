import pytest

from huper.cli import build_parser
from huper.config import load_config
from huper.core.paths import PROJECT_DIR


def test_all_cli_commands_parse_help_free_defaults():
    parser = build_parser()
    for command in (
        "download-models",
        "check",
        "preprocess",
        "train",
        "infer",
        "evaluate",
        "run-all",
    ):
        args = parser.parse_args([command])
        assert args.command == command


def test_paper_config_is_locked():
    config = load_config(PROJECT_DIR / "configs" / "paper.yaml")
    assert config["experiment"]["subjects"] == [1, 2, 5, 7]
    assert config["experiment"]["mqt_num_tokens"] == 256
    assert config["train"]["num_epochs"] == 160
    assert config["train"]["fusion_kl_weight"] == pytest.approx(0.1)
    assert config["paths"]["spice_home"] == str(PROJECT_DIR / "models" / "spice")


def test_source_has_no_machine_specific_paths():
    forbidden = ("/home/", "/Users/", "\\Users\\")
    source_roots = [PROJECT_DIR / "huper", PROJECT_DIR / "llava"]
    for root in source_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{token!r} remains in {path}"
