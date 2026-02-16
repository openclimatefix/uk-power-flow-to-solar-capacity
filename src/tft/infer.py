import gc
import logging
from pathlib import Path
from typing import Any

import torch
from pytorch_forecasting import TimeSeriesDataSet

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from src.tft.model import TFTWithGRU

logger = logging.getLogger(__name__)


def read_test_slice(cfg: DictConfig, limit_sites: list[str] | None = None) -> pd.DataFrame:
    dataset_path = cfg.paths.dataset_path
    m = cfg.model
    freq = pd.Timedelta(minutes=30)
    h_len = int(m.max_prediction_length)

    schema = pq.read_schema(dataset_path)
    available_cols = schema.names

    filters = [("location", "in", limit_sites)] if limit_sites is not None else None

    df = pd.read_parquet(dataset_path, columns=available_cols, filters=filters)

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    df["location"] = df["location"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    test_start = pd.Timestamp(cfg.splits.test_start)
    test_end = pd.Timestamp(cfg.splits.test_end)
    enc_steps = int(m.max_encoder_length)
    t0 = test_start - (freq * enc_steps)
    t1 = test_end + freq * (h_len - 1)

    df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].copy()

    time_idx_col = m.time_idx
    if time_idx_col not in df.columns:
        df[time_idx_col] = df.groupby("location", sort=False).cumcount().astype("int64")
    else:
        df[time_idx_col] = df[time_idx_col].astype("int64")

    return df


def build_datasets(
    cfg: DictConfig,
    df_small: pd.DataFrame,
    model: TFTWithGRU
) -> tuple[TimeSeriesDataSet | None, int]:
    test_start = pd.Timestamp(cfg.splits.test_start)
    ts_col = "timestamp"
    time_idx_col = cfg.model.time_idx

    mask = df_small[ts_col] >= test_start
    if not mask.any():
        return None, -1

    per_loc_min = df_small.loc[mask].groupby("location")[time_idx_col].min()
    min_pred_idx_global = int(per_loc_min.max())

    dataset_params = model.hparams.dataset_parameters.copy()
    dataset_params["min_prediction_idx"] = min_pred_idx_global

    if "predict" in dataset_params:
        del dataset_params["predict"]

    pred_ds = TimeSeriesDataSet.from_parameters(
        dataset_params,
        df_small,
        stop_randomization=True,
    )

    return pred_ds, min_pred_idx_global


def parse_predict_output(result: Any) -> tuple[torch.Tensor, pd.DataFrame]:
    if isinstance(result, dict):
        return result["prediction"], result["index"]

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return result[0], result[1]

    raise RuntimeError("Unsupported prediction output format")


@torch.inference_mode()
def run_predict_one_chunk(
    cfg: DictConfig,
    df_chunk: pd.DataFrame,
    model: TFTWithGRU,
    batch_size: int,
) -> pd.DataFrame:
    pred_ds, min_idx = build_datasets(cfg, df_chunk, model)

    if min_idx == -1 or pred_ds is None or len(pred_ds) == 0:
        return pd.DataFrame()

    pred_loader = pred_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = model.predict(
        pred_loader,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": device, "logger": False}
    )

    preds_t, index = parse_predict_output(result)
    preds = preds_t.detach().cpu().numpy()

    h_len = int(cfg.model.max_prediction_length)
    group_col = cfg.model.group_ids[0]
    time_idx_col = cfg.model.time_idx

    key_map = (
        df_chunk[["location", time_idx_col, "timestamp"]]
        .drop_duplicates(["location", time_idx_col])
        .set_index(["location", time_idx_col])["timestamp"]
        .to_dict()
    )

    rows = []
    for i in range(preds.shape[0]):
        idx_row = index.iloc[i]
        loc = str(idx_row[group_col])
        t0 = int(idx_row[time_idx_col])
        for h in range(h_len):
            ts = key_map.get((loc, t0 + h))
            rows.append({
                "location": loc,
                "timestamp": ts,
                "horizon_step": h + 1,
                "y_hat": float(preds[i, h])
            })

    pred_df = pd.DataFrame(rows).dropna(subset=["timestamp"])
    test_start = pd.Timestamp(cfg.splits.test_start)
    test_end = pd.Timestamp(cfg.splits.test_end)

    pred_df = pred_df[
        (pred_df["timestamp"] >= test_start) & (pred_df["timestamp"] <= test_end)
    ].copy()

    return pred_df


def chunked_predict_and_write(
    cfg: DictConfig,
    all_sites: list[str],
    sites_per_chunk: int,
    model: TFTWithGRU,
    out_path: Path,
    batch_size: int,
) -> None:
    writer = None

    for start in range(0, len(all_sites), sites_per_chunk):
        chunk_sites = all_sites[start : start + sites_per_chunk]
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


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    ckpt_path = Path(cfg.paths.get("inference_ckpt", "production_tft_model.ckpt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFTWithGRU.load_from_checkpoint(str(ckpt_path)).to(device)
    model.eval()

    group_id_col = cfg.model.group_ids[0]
    sites = list(model.hparams.dataset_parameters["categorical_encoders"][group_id_col].classes_)
    sites = sorted([str(s) for s in sites])

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
