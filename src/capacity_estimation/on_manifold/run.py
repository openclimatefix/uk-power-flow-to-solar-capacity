"""On-manifold counterfactual scenario simulation for TFT capacity estimation."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

import hydra
import numpy as np
import pandas as pd
import pyarrow.dataset as ads
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.tft.model import TFTWithGRU
from src.tft.utils import (
    ensure_sorted_and_time_idx,
    fit_calibration,
    get_device,
    model_used_features,
)

from .data_utils import apply_daylight_constants, infer_orientation, pool_min_max
from .model_utils import predict_timeseries
from .sampler import OnManifoldSampler

logger = logging.getLogger(__name__)


def load_calibration_slice(cfg: DictConfig) -> pd.DataFrame:
    """Load validation time slice for fitting the calibration model.

    Args:
        cfg: Full Hydra config.

    Returns:
        Sorted DataFrame with time_idx assigned.
    """
    # y_true ≈ a * y_pred + b
    ds = ads.dataset(cfg.paths.dataset_path, format="parquet")
    ts_col = cfg.splits.timestamp_col
    filt = (ads.field(ts_col) >= pd.Timestamp(cfg.splits.val_start)) & (
        ads.field(ts_col) <= pd.Timestamp(cfg.splits.val_end)
    )
    df = ds.to_table(filter=filt).to_pandas(split_blocks=True, self_destruct=True)
    return ensure_sorted_and_time_idx(df, cfg.model.time_idx)


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for capacity estimation pipeline.

    Args:
        cfg: Hydra config injected automatically.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    torch.set_float32_matmul_precision("high")
    device = get_device()

    out_dir = Path(cfg.get("scenario_output_dir", "scenario_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(cfg.paths.get("inference_ckpt", "production_tft_model.ckpt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading model from %s", ckpt_path)
    model = TFTWithGRU.load_from_checkpoint(str(ckpt_path)).to(device)
    model.eval()

    # Linear Calibration and Bias Correction
    logger.info("Fitting linear calibration on validation slice.")
    cal_df = load_calibration_slice(cfg)
    target_col = cfg.model.target

    val_res = predict_timeseries(cfg, model, cal_df, cfg.get("batch_size", 32))
    a, b = fit_calibration(
        y_true=cal_df[target_col].tolist(),
        y_pred=val_res["y_hat"].tolist(),
    )
    logger.info("Calibration fitted: slope=%.4f, intercept=%.4f", a, b)

    del cal_df, val_res
    gc.collect()

    # Manifold Learning and Latent Space Sampling
    used_features = model_used_features(model)
    sampler_feats = [f for f in cfg.sampler.features if f in used_features]

    sampler = OnManifoldSampler(
        parquet_path=Path(cfg.paths.dataset_path),
        feature_list=sampler_feats,
        device=device,
        save_dir=Path(cfg.get("sampler_ckpt_dir", "sampler_ckpt")),
    )
    sampler.train_or_load()

    # Location / Site Chunking
    group_col = cfg.model.group_ids[0]
    all_sites = sorted(
        str(s) for s in model.hparams.dataset_parameters["categorical_encoders"][group_col].classes_
    )
    sites_per_chunk = cfg.get("sites_per_chunk", 60)
    n_draws = cfg.get("n_draws", 20)

    for i in range(0, len(all_sites), sites_per_chunk):
        chunk_sites = all_sites[i : i + sites_per_chunk]
        logger.info("Processing chunk %d (%d sites)", (i // sites_per_chunk) + 1, len(chunk_sites))

        ds = ads.dataset(cfg.paths.dataset_path, format="parquet")
        df_chunk = ds.to_table(filter=ads.field(group_col).isin(chunk_sites)).to_pandas(
            split_blocks=True, self_destruct=True
        )

        if df_chunk.empty:
            continue

        df_chunk = ensure_sorted_and_time_idx(df_chunk, cfg.model.time_idx)
        df_chunk["timestamp"] = pd.to_datetime(df_chunk[cfg.splits.timestamp_col]).dt.tz_localize(
            None
        )

        per_site_results = []

        for loc, loc_df in tqdm(df_chunk.groupby(group_col), desc="Scenarios", leave=False):
            # Uses orientation_inference logic from config
            orient = infer_orientation(loc_df, cfg)

            # Proxy unseen sites to avoid index errors
            # TODO: Remove as fine tuning on approx 650 locations utilised
            sampler_loc = loc if loc in sampler.loc2id else next(iter(sampler.loc2id.keys()))

            # Counterfactual Scenario Generation
            # month/hour usage constrained by config daylight logic
            candidates = sampler.sample_vectors(
                location=sampler_loc,
                month=int(loc_df["timestamp"].dt.month.median()),
                hour=12,
                k=cfg.get("sampler_k", 64),
            )

            # Partitions candidates using solar_weights and solar_scoring percentiles
            low_pool, high_pool = pool_min_max(candidates, cfg)

            deltas = []
            for _ in range(n_draws):
                v_min = low_pool[np.random.randint(len(low_pool))]
                v_max = high_pool[np.random.randint(len(high_pool))]

                # Update weather and propagate using daylight:hours mask
                df_max = apply_daylight_constants(loc_df, v_max, cfg)
                p_max = predict_timeseries(cfg, model, df_max, cfg.get("batch_size", 32))

                df_min = apply_daylight_constants(loc_df, v_min, cfg)
                p_min = predict_timeseries(cfg, model, df_min, cfg.get("batch_size", 32))

                # Delta = max(0, y_max_cal - y_min_cal)
                y_max = a * p_max["y_hat"] + b
                y_min = a * p_min["y_hat"] + b
                delta = (y_max - y_min) if orient > 0 else (y_min - y_max)
                deltas.extend(delta.clip(lower=0).tolist())

            if deltas:
                per_site_results.append({
                    "location": loc,
                    "mean_impact_mw": np.mean(deltas),
                    "p95_impact_mw": np.percentile(deltas, 95),
                })

        pd.DataFrame(per_site_results).to_csv(out_dir / f"summary_chunk_{i}.csv", index=False)

        del df_chunk
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
