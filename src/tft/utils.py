"""Utility functions for TFT training pipeline."""

from __future__ import annotations

import contextlib
import importlib.metadata
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

_SNAPSHOT_PACKAGES: tuple[str, ...] = (
    "torch",
    "lightning",
    "pytorch-forecasting",
    "hydra-core",
    "pandas",
    "pyarrow",
    "numpy",
)


def intersect_features(
    existing_cols: list[str],
    keys: Iterable[str] | DictConfig | ListConfig | None,
) -> list[str]:
    """Return keys present in existing_cols, warning on missing ones.

    Args:
        existing_cols: Column names in the loaded DataFrame.
        keys: Configured feature names.

    Returns:
        Ordered list of feature names found in both inputs.
    """
    if keys is None:
        return []

    if isinstance(keys, (DictConfig, ListConfig)):
        keys_list: list[str] = [
            str(k) for k in OmegaConf.to_container(keys, resolve=True)
        ]
    else:
        keys_list = [str(k) for k in keys]

    missing = [k for k in keys_list if k not in existing_cols]
    if missing:
        logger.warning(
            "Missing %d configured feature(s): %s", len(missing), missing[:10]
        )

    return [k for k in keys_list if k in existing_cols]


def find_resume_checkpoint(base_dir: Path) -> str | None:
    """Return the most recent checkpoint path under base_dir.

    Args:
        base_dir: Root logging directory.

    Returns:
        Absolute path string, or None if not found.
    """
    if not base_dir.exists():
        return None

    version_dirs: list[tuple[int, Path]] = []
    for path in base_dir.glob("version_*"):
        if path.is_dir() and (match := _VERSION_RE.match(path.name)):
            version_dirs.append((int(match.group(1)), path))

    for _, vdir in sorted(version_dirs, key=lambda t: t[0], reverse=True):
        ckpt_dir = vdir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        last = ckpt_dir / "last.ckpt"
        if last.exists():
            logger.info("Resuming from %s", last)
            return str(last)

        candidates = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            logger.info("Resuming from %s", candidates[0])
            return str(candidates[0])

    return None


def _package_versions() -> dict[str, str]:
    """Return installed versions of key packages.

    Returns:
        Mapping of package name to version string.
    """
    versions: dict[str, str] = {}
    for pkg in _SNAPSHOT_PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[pkg] = importlib.metadata.version(pkg)
    return versions


def save_production_artifacts(
    trainer: pl.Trainer,
    model: torch.nn.Module,
    cfg: DictConfig,
    metrics: dict[str, float],
    dataset: TimeSeriesDataSet,
) -> None:
    """Save production checkpoint and metadata snapshot.

    Args:
        trainer: Fitted Lightning Trainer.
        model: Trained model.
        cfg: Hydra config for this run.
        metrics: Scalar metrics dict.
        dataset: Training TimeSeriesDataSet.
    """
    trainer.save_checkpoint("production_tft_model.ckpt")

    metadata: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "package_versions": _package_versions(),
        "parameters": {
            "max_encoder_length": getattr(dataset, "max_encoder_length", None),
            "max_prediction_length": getattr(dataset, "max_prediction_length", None),
            "target": getattr(dataset, "target", None),
        },
    }
    torch.save(metadata, "production_tft_metadata.pth")
    logger.info("Production artifacts saved successfully.")


def ensure_ts_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with timestamp cast to timezone-naive UTC.

    Args:
        df: Input DataFrame.

    Returns:
        Copy with tz-naive timestamp column, or original if no timestamp column.
    """
    if "timestamp" not in df.columns:
        return df

    df = df.copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    return df


def ensure_sorted_and_time_idx(
    df: pd.DataFrame,
    time_idx_col: str,
) -> pd.DataFrame:
    """Sort df by location/timestamp and assign a per-location time index.

    Args:
        df: Input DataFrame with location and timestamp columns.
        time_idx_col: Column name to create or cast to int64.

    Returns:
        Sorted DataFrame with contiguous integer index per location group.
    """
    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    if time_idx_col not in df.columns:
        df[time_idx_col] = (
            df.groupby("location", sort=False).cumcount().astype("int64")
        )
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
    """Return df copy with time_idx_col as integer steps from a fixed origin.

    Args:
        df: Input DataFrame.
        ts_col: Timestamp column name.
        time_idx_col: Output integer-index column name.
        origin_utc: Reference timestamp in UTC.
        freq_minutes: Step size in minutes.

    Returns:
        Copy of df with time_idx_col added.

    Raises:
        KeyError: If ts_col is absent from df.
    """
    if time_idx_col in df.columns:
        return df

    if ts_col not in df.columns:
        raise KeyError(f"Timestamp column '{ts_col}' missing from dataframe.")

    df = df.copy()
    ts = df[ts_col]

    if getattr(ts.dtype, "tz", None) is not None:
        origin = (
            origin_utc
            if origin_utc.tzinfo is not None
            else origin_utc.tz_localize("UTC")
        )
        delta = ts - origin
    else:
        origin = (
            origin_utc.tz_convert(None)
            if origin_utc.tzinfo is not None
            else origin_utc
        )
        delta = ts - origin

    step_ns = np.int64(freq_minutes) * 60 * 1_000_000_000
    df[time_idx_col] = (
        (delta.dt.total_seconds() * 1_000_000_000).astype("int64") // step_ns
    )
    return df


def get_device() -> str:
    """Return cuda if a GPU is available, otherwise cpu.

    Returns:
        Device string.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_used_features(model: torch.nn.Module) -> set[str]:
    """Extract feature names registered in a model's dataset params.

    Args:
        model: Model with a hparams.dataset_parameters attribute.

    Returns:
        Set of feature-name strings across all feature lists.
    """
    used: set[str] = set()
    dp: dict[str, Any] = getattr(model.hparams, "dataset_parameters", {}) or {}
    feature_keys = (
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "time_varying_known_categoricals",
        "time_varying_unknown_categoricals",
    )
    for key in feature_keys:
        for value in dp.get(key, []) or []:
            if isinstance(value, str):
                used.add(value)
    return used


def parse_predict_output(
    result: object,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Unpack the (predictions, index) pair from model.predict().

    Args:
        result: Raw output from BaseModel.predict().

    Returns:
        Tuple of (predictions tensor, index DataFrame).

    Raises:
        RuntimeError: If result does not match any known output format.
    """
    if isinstance(result, dict):
        preds = result.get("prediction") or result.get("predictions")
        index = result.get("index")
        if torch.is_tensor(preds) and isinstance(index, pd.DataFrame):
            return preds, index

    if (
        isinstance(result, (tuple, list))
        and len(result) >= 2
        and torch.is_tensor(result[0])
        and isinstance(result[1], pd.DataFrame)
    ):
        return result[0], result[1]

    raise RuntimeError(
        f"Could not parse predict() output of type {type(result)}. "
        "Expected a dict with 'prediction'/'index' keys or a "
        "(Tensor, DataFrame) tuple."
    )


def infer_orientation(
    df: pd.DataFrame,
    target_col: str = "active_power_mw_clean",
) -> int:
    """Infer sign relationship between target and solar irradiance.

    Args:
        df: DataFrame with target_col and ssrd_w_m2 columns.
        target_col: Target column name.

    Returns:
        1 if positively correlated with irradiance, -1 otherwise.
    """
    if target_col not in df.columns or "ssrd_w_m2" not in df.columns:
        return 1

    valid = df[[target_col, "ssrd_w_m2"]].dropna()
    if len(valid) < 10:
        return 1

    return 1 if valid[target_col].corr(valid["ssrd_w_m2"]) >= 0 else -1


def fit_calibration(
    y_true: list[float],
    y_pred: list[float],
) -> tuple[float, float]:
    """Fit a linear calibration y_true ≈ a * y_pred + b via least-squares.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Tuple of (slope, intercept).
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 1.0, 0.0

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if np.std(yp) < 1e-6:
        return 1.0, 0.0

    design_matrix = np.column_stack([yp, np.ones_like(yp)])
    coeffs, *_ = np.linalg.lstsq(design_matrix, yt, rcond=None)
    return float(coeffs[0]), float(coeffs[1])
