"""Dataset creation utilities for the TFT production training pipeline."""

from __future__ import annotations

import logging

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

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


def create_production_datasets(
    cfg: DictConfig,
    dataset_path: str,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """Build training and validation TimeSeriesDataSet objects.

    Args:
        cfg: Hydra config with model and splits sub-trees.
        dataset_path: Path to Parquet file or partitioned directory.

    Returns:
        Tuple of (training_dataset, validation_dataset, full_dataframe).
    """
    logger.info("Loading dataset from: %s", dataset_path)

    model_cfg = cfg.model
    needed_cols = list(
        {
            *list(model_cfg.group_ids),
            model_cfg.target,
            "timestamp",
            *list(model_cfg.static_categoricals),
            *list(model_cfg.static_reals),
            *list(model_cfg.time_varying_known_reals),
            *list(model_cfg.time_varying_unknown_reals),
        }
    )

    pdf = _load_parquet_columns(dataset_path, needed_cols)
    pdf = pdf.sort_values(["location", "timestamp"]).reset_index(drop=True)
    pdf["location"] = pdf["location"].astype(str)

    logger.info(
        "Loaded %d rows across %d locations.", len(pdf), pdf["location"].nunique()
    )

    target_col = (
        "active_power_mw_clean"
        if "active_power_mw_clean" in pdf.columns
        else model_cfg.target
    )

    cols = pdf.columns.tolist()
    static_categoricals = intersect_features(cols, model_cfg.static_categoricals)
    static_reals = intersect_features(cols, model_cfg.static_reals)
    known_reals = intersect_features(cols, model_cfg.time_varying_known_reals)
    unknown_reals = intersect_features(cols, model_cfg.time_varying_unknown_reals)
    group_ids = list(model_cfg.group_ids)
    add_target_scales: bool = bool(model_cfg.get("add_target_scales", True))

    pdf["time_idx"] = (
        pdf.groupby("location", sort=False).cumcount().astype("int64")
    )

    shared_kwargs = _shared_tsd_kwargs(
        model_cfg=model_cfg,
        target_col=target_col,
        group_ids=group_ids,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        known_reals=known_reals,
        unknown_reals=unknown_reals,
        add_target_scales=add_target_scales,
    )

    spl = cfg.get("splits", {})
    if spl and spl.get("strategy") == "by_time":
        training_ds, val_ds = _split_by_time(pdf, cfg, shared_kwargs)
    else:
        training_ds, val_ds = _split_by_fraction(pdf, cfg, shared_kwargs)

    return training_ds, val_ds, pdf


def _shared_tsd_kwargs(
    model_cfg: DictConfig,
    target_col: str,
    group_ids: list[str],
    static_categoricals: list[str],
    static_reals: list[str],
    known_reals: list[str],
    unknown_reals: list[str],
    add_target_scales: bool,
) -> dict:
    """Assemble keyword arguments shared by both TimeSeriesDataSet calls.

    Args:
        model_cfg: cfg.model sub-config.
        target_col: Resolved target column name.
        group_ids: Group-ID column names.
        static_categoricals: Static categorical feature names.
        static_reals: Static real feature names.
        known_reals: Time-varying known real feature names.
        unknown_reals: Time-varying unknown real feature names.
        add_target_scales: Whether to append target-scale features.

    Returns:
        Dict of kwargs for TimeSeriesDataSet.
    """
    return {
            "time_idx": "time_idx",
            "target": target_col,
            "group_ids": group_ids,
            "max_encoder_length": model_cfg.max_encoder_length,
            "max_prediction_length": model_cfg.max_prediction_length,
            "static_categoricals": static_categoricals,
            "static_reals": static_reals,
            "time_varying_known_reals": known_reals,
            "time_varying_unknown_reals": unknown_reals,
            "add_relative_time_idx": True,
            "add_target_scales": add_target_scales,
            "add_encoder_length": False,
            "target_normalizer": GroupNormalizer(
                groups=group_ids, transformation="softplus"
            ),
        }


def _split_by_time(
    pdf: pd.DataFrame,
    cfg: DictConfig,
    shared_kwargs: dict,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Create train/val datasets using explicit timestamp boundaries.

    Args:
        pdf: Full sorted DataFrame with time_idx assigned.
        cfg: Hydra config with splits.train_end, splits.val_start, splits.val_end.
        shared_kwargs: Common TimeSeriesDataSet kwargs.

    Returns:
        Tuple of (training_dataset, validation_dataset).
    """
    spl = cfg.splits
    ts_col: str = spl.get("timestamp_col", "timestamp")

    pdf = ensure_ts_naive(pdf)
    train_end = pd.Timestamp(spl["train_end"])
    val_start = pd.Timestamp(spl["val_start"])
    val_end = pd.Timestamp(spl["val_end"])

    val_start_idx = int(
        pdf.loc[pdf[ts_col] >= val_start]
        .groupby("location")["time_idx"]
        .min()
        .max()
    )

    training_dataset = TimeSeriesDataSet(
        pdf[pdf[ts_col] <= train_end].copy(), **shared_kwargs
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        pdf[pdf[ts_col] <= val_end],
        min_prediction_idx=val_start_idx,
        predict=False,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset


def _split_by_fraction(
    pdf: pd.DataFrame,
    cfg: DictConfig,
    shared_kwargs: dict,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Create train/val datasets by fractional time-index cutoffs.

    Args:
        pdf: Full sorted DataFrame with time_idx assigned.
        cfg: Hydra config with train_split and val_split fractions.
        shared_kwargs: Common TimeSeriesDataSet kwargs.

    Returns:
        Tuple of (training_dataset, validation_dataset).
    """
    max_time_idx = int(pdf["time_idx"].max())
    train_cutoff = int(max_time_idx * cfg.train_split)
    val_cutoff = int(max_time_idx * cfg.val_split)

    training_dataset = TimeSeriesDataSet(
        pdf[pdf["time_idx"] <= train_cutoff].copy(), **shared_kwargs
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        pdf[pdf["time_idx"] <= val_cutoff],
        predict=False,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset
