import hashlib
import json
import os

import pytest

from huper import spice_assets
from huper.evaluate import _spice_score_writable


def test_resolve_corenlp_assets_checks_project_local_files(tmp_path, monkeypatch):
    payload = b"standalone-corenlp-test"
    specification = {
        "tiny.jar": {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    monkeypatch.setattr(spice_assets, "CORENLP_JARS", specification)
    config = {"paths": {"spice_home": str(tmp_path)}, "_config_path": "config.yaml"}

    with pytest.raises(RuntimeError, match="download-models"):
        spice_assets.resolve_corenlp_assets(config)

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    jar_path = lib_dir / "tiny.jar"
    jar_path.write_bytes(payload)
    assert spice_assets.resolve_corenlp_assets(config) == {"tiny.jar": jar_path}


def test_spice_uses_explicit_project_corenlp_classpath(tmp_path, monkeypatch):
    lib_dir = tmp_path / "spice" / "lib"
    lib_dir.mkdir(parents=True)
    captured = {}

    def fake_check_call(command, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        output_path = command[command.index("-out") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump([{"image_id": 0, "scores": {"All": {"f": 0.5}}}], handle)

    monkeypatch.setattr("huper.evaluate.subprocess.check_call", fake_check_call)
    score = _spice_score_writable(
        {0: ["reference"]},
        {0: ["candidate"]},
        spice_corenlp_lib=lib_dir,
    )

    assert score == pytest.approx(0.5)
    classpath = captured["command"][captured["command"].index("-cp") + 1]
    assert str(lib_dir / "*") in classpath.split(os.pathsep)
    assert "edu.anu.spice.SpiceScorer" in captured["command"]
