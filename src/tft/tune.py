import gc
import logging
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from omegaconf import DictConfig

from src.tft.model import TFTWithGRU
from src.tft.utils import ensure_time_idx_from_origin

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)


def arrow_time_filter(dataset: ds.Dataset, ts_col: str, start: str, end: str) -> ds.Expression:
    ts_type = dataset.schema.field(ts_col).type
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if pa.types.is_timestamp(ts_type) and ts_type.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")

        start_scalar = pa.scalar(start_ts.to_pydatetime(), type=ts_type)
        end_scalar = pa.scalar(end_ts.to_pydatetime(), type=ts_type)
    else:
        start_scalar = pa.scalar(start_ts.to_pydatetime(), type=ts_type)
        end_scalar = pa.scalar(end_ts.to_pydatetime(), type=ts_type)

    field = ds.field(ts_col)
    return (field >= start_scalar) & (field <= end_scalar)


def arrow_location_exclusion_filter(loc_col: str, exclude_locs: list[str]) -> ds.Expression | None:
    if not exclude_locs:
        return None
    return ~ds.field(loc_col).isin(exclude_locs)


def load_split(
    dataset: ds.Dataset,
    ts_col: str,
    start: str,
    end: str,
    columns: list[str],
    extra_filter: ds.Expression | None = None,
) -> pd.DataFrame:
    filt = arrow_time_filter(dataset, ts_col, start, end)
    if extra_filter is not None:
        filt = filt & extra_filter
    table = dataset.to_table(columns=columns, filter=filt)
    return table.to_pandas(split_blocks=True, self_destruct=True)


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    logger.info("Starting fine-tuning workflow")

    data_path = Path(cfg.paths.dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    splits = cfg.splits
    model_cfg = cfg.model
    train_cfg = cfg.training

    ts_col = splits.timestamp_col
    time_idx_col = model_cfg.time_idx
    target_col = model_cfg.target

    exclude_locs = cfg.get("finetune_exclude_locations", [
        "chatham_west", "wingham", "white_roding_primary"
    ])
    loc_excl_filter = arrow_location_exclusion_filter("location", exclude_locs)

    origin_utc = pd.Timestamp(splits.train_start, tz="UTC")

    arrow_ds = ds.dataset(str(data_path), format="parquet")
    logger.info(f"Timestamp Arrow type: {arrow_ds.schema.field(ts_col).type}")

    schema_names = set(arrow_ds.schema.names)

    cols = (
        [ts_col,
        "location",
        *list(model_cfg.get("static_reals", [])),
        *list(model_cfg.get("time_varying_known_reals",
        [])),
        *list(model_cfg.get("time_varying_unknown_reals",
        [])),
        target_col]
    )

    drop_cols = set(cfg.get("columns_to_drop", []) or [])
    cols = [c for c in cols if c not in drop_cols]
    cols = sorted({c for c in cols if c in schema_names})

    required = {ts_col, "location", target_col}
    missing_required = [c for c in required if c not in cols]
    if missing_required:
        raise RuntimeError(f"Missing required columns in parquet: {missing_required}")

    train_df = load_split(
        arrow_ds, ts_col, splits.train_start, splits.train_end, cols,
        extra_filter=loc_excl_filter,
    )

    train_df["location"] = train_df["location"].astype("category")
    train_df = ensure_time_idx_from_origin(
        train_df,
        ts_col,
        time_idx_col,
        origin_utc,
        freq_minutes=30
    )

    for c in train_df.columns:
        if c not in (ts_col, "location") and pd.api.types.is_float_dtype(train_df[c]):
            train_df[c] = train_df[c].astype("float32")

    if drop_cols:
        train_df = train_df.drop(
            columns=[c for c in drop_cols if c in train_df.columns], errors="ignore"
        )

    logger.info(f"Train rows: {len(train_df)} | Locations: {train_df['location'].nunique()}")

    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx=time_idx_col,
        target=target_col,
        group_ids=list(model_cfg.group_ids),
        max_encoder_length=model_cfg.max_encoder_length,
        min_encoder_length=model_cfg.max_encoder_length,
        max_prediction_length=model_cfg.max_prediction_length,
        static_categoricals=list(model_cfg.get("static_categoricals", [])),
        static_reals=list(model_cfg.get("static_reals", [])),
        time_varying_known_reals=list(model_cfg.time_varying_known_reals),
        time_varying_unknown_reals=list(model_cfg.time_varying_unknown_reals),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
        allow_missing_timesteps=True,
    )

    del train_df
    gc.collect()
    torch.cuda.empty_cache()

    batch_size = int(train_cfg.batch_size) // 2
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    ckpt_path = Path(cfg.paths.get("finetune_ckpt", "production_tft_model.ckpt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info(f"Loading checkpoint: {ckpt_path}")

    model = TFTWithGRU.load_from_checkpoint(str(ckpt_path))
    model.checkpoint_gradient = True
    model.hparams.learning_rate = float(train_cfg.learning_rate) * 0.3

    limit_train_batches = cfg.get("finetune_limit_train_batches", 5000)
    accumulate_grad_batches = cfg.get("finetune_accumulate_grad_batches", 2)

    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            save_weights_only=True,
            filename="finetune-{epoch:02d}",
        ),
    ]

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        limit_train_batches=limit_train_batches,
        max_epochs=train_cfg.max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=float(train_cfg.gradient_clip_val),
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader)
    logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
