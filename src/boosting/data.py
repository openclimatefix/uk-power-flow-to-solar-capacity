"""Dataset creation utilities for XGBoost training pipeline."""

from __future__ import annotations

import logging

import pandas as pd
import pyarrow.dataset as ads
from omegaconf import DictConfig

from src.tft.utils import ensure_ts_naive, intersect_features

logger = logging.getLogger(__name__)


def _load_parquet_columns(
    dataset_path: str,
    needed_cols: list[str],
) -> pd.DataFrame:
    """Read a Parquet dataset via PyArrow, loading only required columns.

    Args:
        dataset_path: File or directory path.
        needed_cols: Desired column names.

    Returns:
        DataFrame with the requested columns.
    """
    ds = ads.dataset(dataset_path, format="parquet")
    available = set(ds.schema.names)
    cols = [c for c in needed_cols if c in available]
    return ds.to_table(columns=cols).to_pandas(split_blocks=True, self_destruct=True)


def create_train_val_split(
    cfg: DictConfig,
    dataset_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load data and return train/val DataFrames with resolved feature list.

    Args:
        cfg: Hydra config with model and splits sub-trees.
        dataset_path: Path to Parquet file or partitioned directory.

    Returns:
        Tuple of (train_df, val_df, feature_cols).
    """
    logger.info("Loading dataset from: %s", dataset_path)

    model_cfg = cfg.model
    splits = cfg.splits
    target = model_cfg.target

    needed_cols = list({
        target,
        "timestamp",
        *list(model_cfg.group_ids),
        *list(model_cfg.features),
    })

    pdf = _load_parquet_columns(dataset_path, needed_cols)
    pdf = ensure_ts_naive(pdf)
    pdf = pdf.sort_values(["location", "timestamp"]).reset_index(drop=True)
    pdf["location"] = pdf["location"].astype(str)

    logger.info("Loaded %d rows across %d locations.", len(pdf), pdf["location"].nunique())

    feature_cols = intersect_features(pdf.columns.tolist(), model_cfg.features)

    train_end = pd.Timestamp(splits.train_end)
    val_start = pd.Timestamp(splits.val_start)
    val_end = pd.Timestamp(splits.val_end)

    train_df = pdf[pdf["timestamp"] <= train_end].copy()
    val_df = pdf[(pdf["timestamp"] >= val_start) & (pdf["timestamp"] <= val_end)].copy()

    logger.info(
        "Train rows: %d | Val rows: %d | Features: %d",
        len(train_df),
        len(val_df),
        len(feature_cols),
    )

    return train_df, val_df, feature_cols
