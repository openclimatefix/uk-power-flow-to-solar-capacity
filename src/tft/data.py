import logging
from pathlib import Path

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

import dask.dataframe as dd
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.tft.utils import ensure_ts_naive, intersect_features

logger = logging.getLogger(__name__)


def create_production_datasets(
    cfg: DictConfig,
    dataset_path: str,
    num_locations: int | None = None,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    logger.info(f"Loading FULL PRODUCTION dataset from: {dataset_path}")

    ddf = dd.read_parquet(
        dataset_path,
        schema="infer",
        split_row_groups=False,
        ignore_metadata_file=True,
    )

    tft_cfg_path = Path("configs/tft/tft_model.yaml")
    try:
        tft_cfg = OmegaConf.load(tft_cfg_path)
        loc_cfg = tft_cfg.get("location_filter", {}) or {}
        mode = loc_cfg.get("mode")
        logger.info(f"Loaded location_filter from {tft_cfg_path} (mode={mode!r})")
    except Exception as e:
        loc_cfg = {}
        mode = None
        logger.warning(f"Failed to load {tft_cfg_path}; proceeding with ALL locations. Error: {e}")

    all_locations = (
        ddf["location"]
        .dropna()
        .astype("string")
        .unique()
        .compute()
        .astype(str)
        .tolist()
    )
    locations = all_locations
    logger.info(f"Training on ALL locations ({len(locations)} locations)")

    ddf_filtered = ddf[ddf["location"].isin(locations)]

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

    available_cols = [c for c in needed_cols if c in ddf_filtered.columns]
    ddf_filtered = ddf_filtered[available_cols]

    logger.info("Computing filtered dataset...")
    pdf = ddf_filtered.compute()
    pdf = pdf.sort_values(["location", "timestamp"]).reset_index(drop=True)
    pdf["location"] = pdf["location"].astype(str)

    target_exists = "active_power_mw_clean" in pdf.columns
    target_col = "active_power_mw_clean" if target_exists else model_cfg.target

    cols = pdf.columns.tolist()
    static_categoricals = intersect_features(cols, model_cfg.static_categoricals)
    static_reals = intersect_features(cols, model_cfg.static_reals)
    known_reals = intersect_features(cols, model_cfg.time_varying_known_reals)
    unknown_reals = intersect_features(cols, model_cfg.time_varying_unknown_reals)

    pdf["time_idx"] = pdf.groupby("location", sort=False).cumcount().astype("int64")

    spl = cfg.get("splits", {})
    group_ids = list(model_cfg.group_ids)

    if spl and spl.get("strategy") == "by_time":
        ts_col = spl.get("timestamp_col", "timestamp")
        pdf = ensure_ts_naive(pdf)

        train_end = pd.Timestamp(spl["train_end"])
        val_start = pd.Timestamp(spl["val_start"])
        val_end = pd.Timestamp(spl["val_end"])

        val_start_idx_per_loc = (
            pdf.loc[pdf[ts_col] >= val_start].groupby("location")["time_idx"].min()
        )
        val_start_idx = int(val_start_idx_per_loc.max())

        training_dataset = TimeSeriesDataSet(
            pdf[pdf[ts_col] <= train_end].copy(),
            time_idx="time_idx",
            target=target_col,
            group_ids=group_ids,
            max_encoder_length=model_cfg.max_encoder_length,
            max_prediction_length=model_cfg.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            pdf[pdf[ts_col] <= val_end],
            min_prediction_idx=val_start_idx,
            predict=False,
            stop_randomization=True,
        )

    else:
        max_time_idx = pdf["time_idx"].max()
        train_cutoff = int(max_time_idx * cfg.train_split)
        val_cutoff = int(max_time_idx * cfg.val_split)

        training_dataset = TimeSeriesDataSet(
            pdf[pdf["time_idx"] <= train_cutoff].copy(),
            time_idx="time_idx",
            target=target_col,
            group_ids=group_ids,
            max_encoder_length=model_cfg.max_encoder_length,
            max_prediction_length=model_cfg.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            pdf[pdf["time_idx"] <= val_cutoff],
            predict=False,
            stop_randomization=True,
        )

    return training_dataset, validation_dataset, pdf
