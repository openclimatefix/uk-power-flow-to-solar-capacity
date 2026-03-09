"""Fine-tuning pipeline for an existing TFT checkpoint."""

from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ads
from omegaconf import DictConfig

from src.tft.model import TFTWithGRU

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

_DEFAULT_EXCLUDE_LOCATIONS: tuple[str, ...] = (
    "chatham_west",
    "wingham",
    "white_roding_primary",
)


def build_arrow_filter(
    dataset: ads.Dataset,
    ts_col: str,
    start: str,
    end: str,
    loc_col: str | None = None,
    exclude_locs: list[str] | None = None,
) -> ads.Expression:
    """Build a combined Arrow filter for a time range and optional exclusions.

    Args:
        dataset: PyArrow dataset defining the timestamp type.
        ts_col: Timestamp column name.
        start: ISO-8601 start timestamp (inclusive).
        end: ISO-8601 end timestamp (inclusive).
        loc_col: Location column name for exclusion filtering.
        exclude_locs: Location values to exclude.

    Returns:
        Compound pyarrow dataset Expression.
    """
    ts_type = dataset.schema.field(ts_col).type

    def _to_scalar(ts_str: str) -> pa.Scalar:
        ts = pd.Timestamp(ts_str)
        if pa.types.is_timestamp(ts_type) and ts_type.tz is not None:
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return pa.scalar(ts.to_pydatetime(), type=ts_type)

    filt = (ads.field(ts_col) >= _to_scalar(start)) & (ads.field(ts_col) <= _to_scalar(end))

    if loc_col and exclude_locs:
        filt = filt & ~ads.field(loc_col).isin(exclude_locs)

    return filt


def load_split(
    dataset: ads.Dataset,
    ts_col: str,
    start: str,
    end: str,
    columns: list[str],
    loc_col: str | None = None,
    exclude_locs: list[str] | None = None,
) -> pd.DataFrame:
    """Read a filtered time slice from dataset into a DataFrame.

    Args:
        dataset: Source PyArrow dataset.
        ts_col: Timestamp column name.
        start: Start of time window (inclusive, ISO-8601).
        end: End of time window (inclusive, ISO-8601).
        columns: Columns to materialise.
        loc_col: Location column name for exclusion filtering.
        exclude_locs: Location values to exclude.

    Returns:
        DataFrame with requested columns and filters applied.
    """
    filt = build_arrow_filter(
        dataset, ts_col, start, end, loc_col=loc_col, exclude_locs=exclude_locs
    )
    table = dataset.to_table(columns=columns, filter=filt)
    return table.to_pandas(split_blocks=True, self_destruct=True)


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the fine-tuning pipeline.

    Args:
        cfg: Hydra config injected automatically.

    Raises:
        FileNotFoundError: If dataset or checkpoint path does not exist.
        RuntimeError: If required columns are absent from the Parquet schema.
    """
    torch.set_float32_matmul_precision("high")
    logger.info("Starting fine-tuning workflow.")

    data_path = Path(cfg.paths.dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    splits = cfg.splits
    model_cfg = cfg.model
    train_cfg = cfg.training

    ts_col: str = splits.timestamp_col
    time_idx_col: str = model_cfg.time_idx
    target_col: str = model_cfg.target

    exclude_locs: list[str] = list(
        cfg.get("finetune_exclude_locations", list(_DEFAULT_EXCLUDE_LOCATIONS))
    )
    drop_cols: set[str] = set(cfg.get("columns_to_drop", []) or [])

    arrow_ds = ads.dataset(str(data_path), format="parquet")
    logger.info("Timestamp Arrow type: %s", arrow_ds.schema.field(ts_col).type)

    schema_names = set(arrow_ds.schema.names)

    raw_cols = [
        ts_col,
        "location",
        *model_cfg.get("static_reals", []),
        *model_cfg.get("time_varying_known_reals", []),
        *model_cfg.get("time_varying_unknown_reals", []),
        target_col,
    ]

    cols = sorted({c for c in raw_cols if c in schema_names and c not in drop_cols})

    missing_required = [c for c in (ts_col, "location", target_col) if c not in cols]
    if missing_required:
        raise RuntimeError(f"Missing required columns in parquet: {missing_required}")

    train_df = load_split(
        arrow_ds,
        ts_col,
        splits.train_start,
        splits.train_end,
        cols,
        loc_col="location",
        exclude_locs=exclude_locs,
    )

    train_df["location"] = train_df["location"].astype("category")
    train_df = train_df.sort_values(["location", ts_col]).reset_index(drop=True)
    train_df[time_idx_col] = train_df.groupby("location", sort=False).cumcount().astype("int64")

    for col in train_df.select_dtypes("float64").columns:
        train_df[col] = train_df[col].astype("float32")

    logger.info(
        "Train rows: %d | Locations: %d",
        len(train_df),
        train_df["location"].nunique(),
    )

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
    logger.info("Loading checkpoint: %s", ckpt_path)

    model = TFTWithGRU.load_from_checkpoint(str(ckpt_path))

    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    else:
        model.checkpoint_gradient = True

    model.hparams.learning_rate = float(train_cfg.learning_rate) * 0.3

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="bf16-mixed",
        limit_train_batches=int(cfg.get("finetune_limit_train_batches", 5000)),
        max_epochs=train_cfg.max_epochs,
        accumulate_grad_batches=int(cfg.get("finetune_accumulate_grad_batches", 2)),
        gradient_clip_val=float(train_cfg.gradient_clip_val),
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                every_n_epochs=1,
                save_weights_only=True,
                filename="finetune-{epoch:02d}",
            ),
        ],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        logger=CSVLogger(save_dir="logs", name="tft_finetune"),
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader)
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
