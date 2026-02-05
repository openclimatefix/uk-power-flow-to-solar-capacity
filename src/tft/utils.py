import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet

logger = logging.getLogger(__name__)


def intersect_features(
    existing_cols: list[str],
    keys: Optional[Union[Iterable[str], DictConfig, ListConfig]],
) -> list[str]:
    if keys is None:
        return []

    if isinstance(keys, (DictConfig, ListConfig)):
        keys_list: list[str] = [
            str(k) for k in OmegaConf.to_container(keys, resolve=True)  # type: ignore
        ]
    else:
        keys_list = [str(k) for k in keys]

    miss = [k for k in keys_list if k not in existing_cols]
    if miss:
        logger.warning(f"Missing {len(miss)} configured features: {miss[:10]}")

    return [k for k in keys_list if k in existing_cols]


def find_resume_checkpoint(base_dir: Path) -> Optional[str]:
    if not base_dir.exists():
        return None

    try:
        version_dirs = sorted(
            base_dir.glob("version_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        version_dirs = []

    for vdir in version_dirs:
        ckpt_dir = vdir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last)

        candidates = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    return None


def save_production_artifacts(
    trainer: pl.Trainer,
    model: torch.nn.Module,
    cfg: DictConfig,
    metrics: dict[str, float],
    dataset: TimeSeriesDataSet,
) -> None:
    trainer.save_checkpoint("production_tft_model.ckpt")

    metadata = {
        "state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "parameters": {
            "max_encoder_length": getattr(dataset, "max_encoder_length", None),
            "max_prediction_length": getattr(dataset, "max_prediction_length", None),
            "target": getattr(dataset, "target", None),
        },
    }
    torch.save(metadata, "production_tft_metadata.pth")
    logger.info("Production artifacts saved successfully.")
