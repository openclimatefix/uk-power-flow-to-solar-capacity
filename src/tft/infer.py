"""Chunked inference pipeline for the TFT production model."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from pytorch_forecasting import TimeSeriesDataSet

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ads
import pyarrow.parquet as pq
from omegaconf import DictConfig

from src.tft.model import TFTWithGRU
from src.tft.utils import get_device, parse_predict_output

logger = logging.getLogger(__name__)


def read_test_slice(
    cfg: DictConfig,
    limit_sites: list[str] | None = None,
) -> pd.DataFrame:
    """Load the encoder + horizon window for the configured test period.

    Args:
        cfg: Full Hydra config.
        limit_sites: Optional location values to filter on.

    Returns:
        DataFrame sorted by (location, timestamp) with a time_idx column.
    """
    model_cfg = cfg.model
    freq = pd.Timedelta(minutes=30)
    h_len = int(model_cfg.max_prediction_length)
    enc_steps = int(model_cfg.max_encoder_length)

    ds = ads.dataset(cfg.paths.dataset_path, format="parquet")

    filt: ads.Expression | None = None
    if limit_sites is not None:
        filt = ads.field("location").isin(limit_sites)

    table = ds.to_table(columns=ds.schema.names, filter=filt)
    df = table.to_pandas(split_blocks=True, self_destruct=True)

    if df.empty:
        return df

    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    df["location"] = df["location"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    test_start = pd.Timestamp(cfg.splits.test_start)
    test_end = pd.Timestamp(cfg.splits.test_end)
    window_start = test_start - freq * enc_steps
    window_end = test_end + freq * (h_len - 1)

    df = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)].copy()

    time_idx_col = model_cfg.time_idx
    if time_idx_col not in df.columns:
        df[time_idx_col] = df.groupby("location", sort=False).cumcount().astype("int64")
    else:
        df[time_idx_col] = df[time_idx_col].astype("int64")

    return df


def build_pred_dataset(
    cfg: DictConfig,
    df: pd.DataFrame,
    model: TFTWithGRU,
) -> tuple[TimeSeriesDataSet | None, int]:
    """Construct a prediction TimeSeriesDataSet from df.

    Args:
        cfg: Full Hydra config.
        df: DataFrame covering encoder window plus test horizon.
        model: Loaded TFTWithGRU instance.

    Returns:
        Tuple of (dataset, min_pred_idx), or (None, -1) if no rows in test window.
    """
    test_start = pd.Timestamp(cfg.splits.test_start)
    time_idx_col = cfg.model.time_idx

    mask = df["timestamp"] >= test_start
    if not mask.any():
        return None, -1

    min_pred_idx = int(df.loc[mask].groupby("location")[time_idx_col].min().max())

    dataset_params = dict(model.hparams.dataset_parameters)
    dataset_params["min_prediction_idx"] = min_pred_idx
    dataset_params.pop("predict", None)

    pred_ds = TimeSeriesDataSet.from_parameters(dataset_params, df, stop_randomization=True)
    return pred_ds, min_pred_idx


@torch.inference_mode()
def run_predict_one_chunk(
    cfg: DictConfig,
    df_chunk: pd.DataFrame,
    model: TFTWithGRU,
    batch_size: int,
) -> pd.DataFrame:
    """Run inference over a single site chunk.

    Args:
        cfg: Full Hydra config.
        df_chunk: Encoder + horizon data for the chunk's sites.
        model: Loaded TFTWithGRU in eval mode.
        batch_size: Batch size for the prediction dataloader.

    Returns:
        DataFrame with columns (location, timestamp, horizon_step, y_hat),
        filtered to [test_start, test_end]. Empty if no predictions produced.
    """
    pred_ds, min_idx = build_pred_dataset(cfg, df_chunk, model)

    if min_idx == -1 or pred_ds is None or len(pred_ds) == 0:
        return pd.DataFrame()

    pred_loader = pred_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, shuffle=False
    )

    result = model.predict(
        pred_loader,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": get_device(), "logger": False},
    )

    preds_t, index = parse_predict_output(result)
    preds = preds_t.detach().cpu().numpy()

    h_len = int(cfg.model.max_prediction_length)
    group_col = cfg.model.group_ids[0]
    time_idx_col = cfg.model.time_idx

    key_map: dict[tuple[str, int], pd.Timestamp] = (
        df_chunk[["location", time_idx_col, "timestamp"]]
        .drop_duplicates(["location", time_idx_col])
        .set_index(["location", time_idx_col])["timestamp"]
        .to_dict()
    )

    n_samples = preds.shape[0]
    locs = index[group_col].astype(str).to_numpy()
    t0s = index[time_idx_col].to_numpy().astype(int)

    locs_rep = np.repeat(locs, h_len)
    t0s_rep = np.repeat(t0s, h_len)
    horizon_offsets = np.tile(np.arange(h_len), n_samples)
    tidxs = t0s_rep + horizon_offsets

    timestamps = [key_map.get((loc, int(tidx))) for loc, tidx in zip(locs_rep, tidxs, strict=True)]

    pred_df = pd.DataFrame({
        "location": locs_rep,
        "timestamp": timestamps,
        "horizon_step": horizon_offsets + 1,
        "y_hat": preds.ravel(),
    }).dropna(subset=["timestamp"])

    test_start = pd.Timestamp(cfg.splits.test_start)
    test_end = pd.Timestamp(cfg.splits.test_end)
    in_window = (pred_df["timestamp"] >= test_start) & (pred_df["timestamp"] <= test_end)
    return pred_df.loc[in_window].copy()


def chunked_predict_and_write(
    cfg: DictConfig,
    all_sites: list[str],
    sites_per_chunk: int,
    model: TFTWithGRU,
    out_path: Path,
    batch_size: int,
) -> None:
    """Iterate over site chunks, run inference, and stream results to Parquet.

    Args:
        cfg: Full Hydra config.
        all_sites: Complete ordered list of site identifiers.
        sites_per_chunk: Number of sites per iteration.
        model: Loaded TFTWithGRU in eval mode.
        out_path: Destination Parquet file path.
        batch_size: Batch size for each prediction dataloader.
    """
    writer: pq.ParquetWriter | None = None
    n_chunks = (len(all_sites) + sites_per_chunk - 1) // sites_per_chunk

    for chunk_idx, start in enumerate(range(0, len(all_sites), sites_per_chunk)):
        chunk_sites = all_sites[start : start + sites_per_chunk]
        logger.info("Chunk %d/%d — %d sites.", chunk_idx + 1, n_chunks, len(chunk_sites))

        df_chunk = read_test_slice(cfg, limit_sites=chunk_sites)
        if df_chunk.empty:
            continue

        pred_df = run_predict_one_chunk(cfg, df_chunk, model, batch_size)
        if pred_df.empty:
            continue

        table = pa.Table.from_pandas(pred_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        gc.collect()

    if writer:
        writer.close()
        logger.info("Predictions written to %s", out_path)
    else:
        logger.warning("No predictions produced — output file not written.")


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the inference pipeline.

    Args:
        cfg: Hydra config injected automatically.

    Raises:
        FileNotFoundError: If the specified checkpoint does not exist.
    """
    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(cfg.paths.get("inference_ckpt", "production_tft_model.ckpt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    model: TFTWithGRU = TFTWithGRU.load_from_checkpoint(str(ckpt_path)).to(device)
    model.eval()

    group_id_col = cfg.model.group_ids[0]
    sites = sorted(
        str(s)
        for s in model.hparams.dataset_parameters["categorical_encoders"][group_id_col].classes_
    )
    logger.info("Running inference over %d sites.", len(sites))

    out_path = Path(cfg.paths.get("output_path", "tft_predictions.parquet"))

    chunked_predict_and_write(
        cfg=cfg,
        all_sites=sites,
        sites_per_chunk=cfg.get("sites_per_chunk", 10),
        model=model,
        out_path=out_path,
        batch_size=cfg.get("batch_size", 4),
    )


if __name__ == "__main__":
    main()
