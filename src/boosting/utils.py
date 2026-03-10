"""Utility functions for the XGBoost training pipeline."""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_SNAPSHOT_PACKAGES: tuple[str, ...] = (
    "xgboost",
    "pandas",
    "pyarrow",
    "numpy",
    "hydra-core",
    "scikit-learn",
)


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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and SMAPE for regression predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dict with mae, rmse, and smape keys.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}


def save_xgboost_artifacts(
    model: xgb.XGBRegressor,
    cfg: DictConfig,
    metrics: dict[str, float],
    feature_cols: list[str],
    output_dir: str = ".",
) -> None:
    """Save XGBoost model and metadata snapshot.

    Args:
        model: Trained XGBRegressor.
        cfg: Hydra config for this run.
        metrics: Scalar metrics dict.
        feature_cols: Feature columns used during training.
        output_dir: Directory to write artifacts into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "production_xgboost_model.json"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    import json

    metadata: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "feature_cols": feature_cols,
        "package_versions": _package_versions(),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
    }
    meta_path = out / "production_xgboost_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


def log_feature_importance(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    top_n: int = 20,
) -> None:
    """Log top-N features by importance score.

    Args:
        model: Trained XGBRegressor.
        feature_cols: Feature column names in training order.
        top_n: Number of top features to log.
    """
    scores = model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info("Top %d features:\n%s", top_n, importance_df.to_string(index=False))
