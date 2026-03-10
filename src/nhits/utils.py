"""Utility functions for the NHiTS training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
import torch

from src.tft.utils import find_resume_checkpoint, save_production_artifacts

logger = logging.getLogger(__name__)

__all__ = [
    "find_resume_checkpoint",
    "get_device",
    "save_production_artifacts",
]


def get_device() -> str:
    """Return cuda if a GPU is available, otherwise cpu.

    Returns:
        Device string.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_nhits_resume_checkpoint(base_dir: Path) -> str | None:
    """Return the most recent NHiTS checkpoint under base_dir.

    Args:
        base_dir: Root logging directory for NHiTS runs.

    Returns:
        Absolute path string, or None if not found.
    """
    return find_resume_checkpoint(base_dir)


def save_nhits_artifacts(
    trainer: pl.Trainer,
    model: torch.nn.Module,
    cfg: object,
    metrics: dict[str, float],
    dataset: object,
) -> None:
    """Save NHiTS production checkpoint and metadata snapshot.

    Args:
        trainer: Fitted Lightning Trainer.
        model: Trained NHiTS model.
        cfg: Hydra config for this run.
        metrics: Scalar metrics dict.
        dataset: Training TimeSeriesDataSet.
    """
    save_production_artifacts(
        trainer=trainer,
        model=model,
        cfg=cfg,
        metrics=metrics,
        dataset=dataset,
    )
