"""Tests for src.tft.train and find_resume_checkpoint."""

from __future__ import annotations

import os
import time

import torch

from omegaconf import OmegaConf

from src.tft.utils import find_resume_checkpoint


def test_find_resume_checkpoint_prefers_last(tmp_path) -> None:
    version_dir = tmp_path / "version_0" / "checkpoints"
    version_dir.mkdir(parents=True)
    (version_dir / "last.ckpt").write_bytes(b"dummy")
    (version_dir / "tft-epoch=01-val_loss=0.1.ckpt").write_bytes(b"dummy2")

    result = find_resume_checkpoint(tmp_path)
    assert result is not None
    assert result.endswith("last.ckpt")


def test_find_resume_checkpoint_returns_none_when_missing(tmp_path) -> None:
    assert find_resume_checkpoint(tmp_path / "nonexistent") is None


def test_find_resume_checkpoint_picks_newest_version(tmp_path) -> None:
    for v in ("version_0", "version_1"):
        d = tmp_path / v / "checkpoints"
        d.mkdir(parents=True)
        (d / "last.ckpt").write_bytes(b"ckpt")

    os.utime(tmp_path / "version_1", (time.time() + 100, time.time() + 100))

    result = find_resume_checkpoint(tmp_path)
    assert result is not None
    assert "version_1" in result


def test_hparams_compatibility_logic(monkeypatch, tmp_path) -> None:
    cfg = OmegaConf.create({"training": {"learning_rate": 0.001}})
    dummy_path = tmp_path / "test.ckpt"

    monkeypatch.setattr(
        "torch.load",
        lambda *_, **__: {"hyper_parameters": {"learning_rate": 0.001}}
    )

    ckpt = torch.load(str(dummy_path))
    assert ckpt["hyper_parameters"]["learning_rate"] == cfg.training.learning_rate
