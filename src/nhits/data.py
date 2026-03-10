"""Dataset creation utilities for the NHiTS training pipeline."""

from __future__ import annotations

import logging

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder

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
    """Build training and validation TimeSeriesDataSet objects for NHiTS.

    Args:
        cfg: Hydra config with model and splits sub-trees.
        dataset_path: Path to Parquet file or partitioned directory.

    Returns:
        Tuple of (training_dataset, validation_dataset, full_dataframe).
    """
    logger.info("Loading dataset from: %s", dataset_path)

    model_cfg = cfg.model
    splits = cfg.splits

    needed_cols = list({
        *list(model_cfg.group_ids),
        model_cfg.target,
        "timestamp",
        *list(model_cfg.get("static_categoricals", [])),
        *list(model_cfg.get("static_reals", [])),
        *list(model_cfg.get("time_varying_known_reals", [])),
        *list(model_cfg.get("time_varying_unknown_reals", [])),
    })

    pdf = _load_parquet_columns(dataset_path, needed_cols)
    pdf = pdf.sort_values(["location", "timestamp"]).reset_index(drop=True)
    pdf["location"] = pdf["location"].astype(str)

    logger.info("Loaded %d rows across %d locations.", len(pdf), pdf["location"].nunique())

    pdf = ensure_ts_naive(pdf)

    t0 = pd.Timestamp(splits.train_start).tz_localize(None)
    pdf["time_idx"] = ((pdf["timestamp"] - t0) / pd.Timedelta(minutes=30)).astype(int)
    pdf = pdf.sort_values(["location", "time_idx"]).reset_index(drop=True)

    cols = pdf.columns.tolist()
    static_categoricals = intersect_features(cols, model_cfg.get("static_categoricals", []))
    static_reals = intersect_features(cols, model_cfg.get("static_reals", []))
    known_reals = intersect_features(cols, model_cfg.get("time_varying_known_reals", []))
    unknown_reals = intersect_features(cols, model_cfg.get("time_varying_unknown_reals", []))

    train_end = pd.Timestamp(splits.train_end)
    val_end = pd.Timestamp(splits.val_end)

    training_ds = TimeSeriesDataSet(
        pdf[pdf["timestamp"] <= train_end].copy(),
        time_idx="time_idx",
        target=model_cfg.target,
        group_ids=list(model_cfg.group_ids),
        max_encoder_length=int(model_cfg.max_encoder_length),
        max_prediction_length=int(model_cfg.max_prediction_length),
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=list(model_cfg.group_ids), transformation="softplus"
        ),
        categorical_encoders={"location": NaNLabelEncoder(add_nan=True)},
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        pdf[pdf["timestamp"] <= val_end],
        stop_randomization=True,
    )

    logger.info(
        "Training samples: %d | Validation samples: %d",
        len(training_ds),
        len(validation_ds),
    )

    return training_ds, validation_ds, pdf
