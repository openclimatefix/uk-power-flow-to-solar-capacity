import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)
_VERSION_RE = re.compile(r"^version_(\d+)$")


def intersect_features(
    existing_cols: list[str],
    keys: Iterable[str] | DictConfig | ListConfig | None,
) -> list[str]:
    if keys is None:
        return []

    if isinstance(keys, (DictConfig, ListConfig)):
        keys_list: list[str] = [
            str(k) for k in OmegaConf.to_container(keys, resolve=True)
        ]
    else:
        keys_list = [str(k) for k in keys]

    miss = [k for k in keys_list if k not in existing_cols]
    if miss:
        logger.warning(f"Missing {len(miss)} configured features: {miss[:10]}")

    return [k for k in keys_list if k in existing_cols]


def find_resume_checkpoint(base_dir: Path) -> str | None:
    if not base_dir.exists():
        return None

    version_dirs: list[Path] = []
    for p in base_dir.glob("version_*"):
        if not p.is_dir():
            continue
        m = _VERSION_RE.match(p.name)
        if m:
            version_dirs.append(p)

    version_dirs.sort(key=lambda p: int(_VERSION_RE.match(p.name).group(1)), reverse=True)

    for vdir in version_dirs:
        ckpt_dir = vdir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last)

        try:
            candidates = sorted(
                ckpt_dir.glob("*.ckpt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            candidates = []

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


def ensure_ts_naive(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    return df


def ensure_sorted_and_time_idx(df: pd.DataFrame, time_idx_col: str) -> pd.DataFrame:
    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    if time_idx_col not in df.columns:
        df[time_idx_col] = df.groupby("location", sort=False).cumcount().astype("int64")
    else:
        df[time_idx_col] = df[time_idx_col].astype("int64")
    return df


def ensure_time_idx_from_origin(
    df: pd.DataFrame,
    ts_col: str,
    time_idx_col: str,
    origin_utc: pd.Timestamp,
    freq_minutes: int = 30,
) -> pd.DataFrame:
    """Compute time_idx as integer steps from a fixed UTC origin timestamp.

    Unlike ensure_sorted_and_time_idx (which uses per-location cumcount),
    this produces a globally consistent index based on wall-clock time,
    suitable for fine-tuning where the origin must match the pre-trained model.
    """
    if time_idx_col in df.columns:
        return df

    if ts_col not in df.columns:
        raise KeyError(f"Timestamp column '{ts_col}' missing from dataframe.")

    ts = df[ts_col]

    if getattr(ts.dtype, "tz", None) is not None:
        if origin_utc.tzinfo is None:
            origin_utc = origin_utc.tz_localize("UTC")
        else:
            origin_utc = origin_utc.tz_convert("UTC")
        delta = ts - origin_utc
    else:
        origin_naive = origin_utc.tz_convert(None) if origin_utc.tzinfo is not None else origin_utc
        delta = ts - origin_naive

    step_ns = np.int64(freq_minutes) * 60 * 1_000_000_000
    df[time_idx_col] = (delta.dt.total_seconds() * 1_000_000_000).astype("int64") // step_ns
    return df


def model_used_features(model: torch.nn.Module) -> set[str]:
    used: set[str] = set()
    dp = getattr(model.hparams, "dataset_parameters", {})
    keys = [
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "time_varying_known_categoricals",
        "time_varying_unknown_categoricals",
    ]
    for k in keys:
        for v in dp.get(k, []) or []:
            if isinstance(v, str):
                used.add(v)
    return used


def parse_predict_output(result: Any) -> tuple[torch.Tensor, pd.DataFrame]:
    if isinstance(result, dict):
        preds = result.get("prediction") or result.get("predictions")
        index = result.get("index")
        if preds is not None and torch.is_tensor(preds):
            return preds, index

    if isinstance(result, (tuple, list)) and len(result) >= 2 and torch.is_tensor(result[0]):
        return result[0], result[1] if isinstance(result[1], pd.DataFrame) else None

    raise RuntimeError(f"Could not parse predict() output of type {type(result)}")


def infer_orientation(df: pd.DataFrame, target_col: str = "active_power_mw_clean") -> int:
    if target_col not in df.columns or "ssrd_w_m2" not in df.columns:
        return 1

    valid = df[[target_col, "ssrd_w_m2"]].dropna()
    if len(valid) < 10:
        return 1

    corr = valid[target_col].corr(valid["ssrd_w_m2"])
    return 1 if corr >= 0 else -1


def fit_calibration(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 1.0, 0.0

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if np.std(y_pred_arr) < 1e-6:
        return 1.0, 0.0

    a = np.cov(y_true_arr, y_pred_arr)[0, 1] / (np.var(y_pred_arr) + 1e-8)
    b = np.mean(y_true_arr) - a * np.mean(y_pred_arr)
    return float(a), float(b)
