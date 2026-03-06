"""
Brute force scenario simulation using TFT.

This method estimates potential PV capacity.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

import hydra
import pandas as pd
import pyarrow.dataset as ads
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.tft.model import TFTWithGRU
from src.tft.utils import ensure_sorted_and_time_idx, get_device

from .brute_force_utils import apply_scenario, build_coherent_mods
from .model_utils import predict_timeseries

logger = logging.getLogger(__name__)


def proxy_for_unseen(loc: str, trained: list[str]) -> str:
    """Map unseen locations to trained encoder labels."""
    return trained[hash(loc) % len(trained)]


@hydra.main(
    version_base=None,
    config_path="../../configs/scenarios",
    config_name="brute_force",
)
def main(cfg: DictConfig) -> None:

    torch.set_float32_matmul_precision("high")

    device = get_device()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(cfg.paths.inference_ckpt)

    logger.info("Loading TFT model: %s", ckpt)

    model = TFTWithGRU.load_from_checkpoint(str(ckpt)).to(device)
    model.eval()

    group_col = cfg.model.group_ids[0]
    time_idx = cfg.model.time_idx

    enc = model.hparams.dataset_parameters["categorical_encoders"][group_col]
    trained_sites = list(map(str, enc.classes_))

    ds = ads.dataset(cfg.paths.dataset_path, format="parquet")

    all_sites = sorted(
        ds.to_table(columns=[group_col])
        .to_pandas()[group_col]
        .astype(str)
        .unique()
    )

    sites_per_chunk = cfg.get("sites_per_chunk", 50)

    for i in range(0, len(all_sites), sites_per_chunk):

        chunk = all_sites[i:i + sites_per_chunk]

        logger.info("Processing %d sites", len(chunk))

        df = (
            ds.to_table(filter=ads.field(group_col).isin(chunk))
            .to_pandas(split_blocks=True)
        )

        if df.empty:
            continue

        df = ensure_sorted_and_time_idx(df, time_idx, group_col)

        results = []

        for loc, loc_df in tqdm(df.groupby(group_col), desc="sites"):

            proxy = loc if loc in trained_sites else proxy_for_unseen(loc, trained_sites)

            work = loc_df.copy()
            work[group_col] = proxy

            feature_cols = [
                f for f in cfg.scenario_features if f in work.columns
            ]

            if not feature_cols:
                continue

            mins, maxs = build_coherent_mods(
                work,
                feature_cols,
                cfg.quantiles.high,
                cfg.quantiles.low,
                cfg.push.high,
                cfg.push.low,
            )

            df_max = apply_scenario(work, maxs)
            df_min = apply_scenario(work, mins)

            p_max = predict_timeseries(cfg, model, df_max, cfg.batch_size)
            p_min = predict_timeseries(cfg, model, df_min, cfg.batch_size)

            if p_max.empty or p_min.empty:
                continue

            merged = p_max.merge(
                p_min,
                on=["location", "timestamp", "horizon_step"],
                suffixes=("_max", "_min"),
            )

            merged["delta"] = (merged["y_hat_max"] - merged["y_hat_min"]).clip(lower=0)

            results.append(
                {
                    "location": loc,
                    "mean_impact_mw": float(merged["delta"].mean()),
                    "p95_impact_mw": float(merged["delta"].quantile(0.95)),
                }
            )

        pd.DataFrame(results).to_csv(
            out_dir / f"summary_chunk_{i}.csv",
            index=False,
        )

        del df
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
