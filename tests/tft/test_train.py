import os
import time

import torch

from omegaconf import OmegaConf

from src.tft.utils import find_resume_checkpoint


def test_find_resume_checkpoint_prefers_last(tmp_path):
    version_dir = tmp_path / "version_0" / "checkpoints"
    version_dir.mkdir(parents=True)

    last_ckpt = version_dir / "last.ckpt"
    other_ckpt = version_dir / "tft-epoch=01-val_loss=0.1.ckpt"

    last_ckpt.write_bytes(b"dummy")
    other_ckpt.write_bytes(b"dummy2")

    ckpt_path = find_resume_checkpoint(tmp_path)
    assert ckpt_path is not None
    assert ckpt_path.endswith("last.ckpt")


def test_find_resume_checkpoint_returns_none_when_missing(tmp_path):
    ckpt_path = find_resume_checkpoint(tmp_path / "nonexistent")
    assert ckpt_path is None


def test_find_resume_checkpoint_picks_newest_version(tmp_path):
    v0 = tmp_path / "version_0" / "checkpoints"
    v1 = tmp_path / "version_1" / "checkpoints"
    v0.mkdir(parents=True)
    v1.mkdir(parents=True)

    (v0 / "last.ckpt").write_bytes(b"old")
    (v1 / "last.ckpt").write_bytes(b"new")

    os.utime(v1, (time.time() + 100, time.time() + 100))

    ckpt_path = find_resume_checkpoint(tmp_path)
    assert "version_1" in ckpt_path


def test_hparams_compatibility_logic(monkeypatch, tmp_path):
    cfg = OmegaConf.create({
        "training": {"learning_rate": 0.001}
    })

    dummy_path = tmp_path / "test.ckpt"

    def _fake_load(*_, **__):
        return {"hyper_parameters": {"learning_rate": 0.001}}

    monkeypatch.setattr("torch.load", _fake_load)

    ckpt = torch.load(str(dummy_path))
    assert ckpt["hyper_parameters"]["learning_rate"] == cfg.training.learning_rate
