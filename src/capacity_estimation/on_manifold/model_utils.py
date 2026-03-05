"""Model utility functions for TFT capacity estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

DEFAULT_TFT_CONFIG_PATH = Path("configs/tft/tft_model.yaml")
_TFT_CFG_CACHE: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping/dict: {path}")
    return cfg


def get_tft_cfg(config_path: Path = DEFAULT_TFT_CONFIG_PATH) -> dict[str, Any]:
    """Return cached TFT config, loading from disk on first call.

    Args:
        config_path: Path to tft_model.yaml.

    Returns:
        Parsed config dict.
    """
    global _TFT_CFG_CACHE
    if _TFT_CFG_CACHE is None:
        _TFT_CFG_CACHE = _load_yaml(config_path)
        logger.info("Loaded TFT config from %s", config_path)
    return _TFT_CFG_CACHE


def get_tft_fields(config_path: Path = DEFAULT_TFT_CONFIG_PATH) -> dict[str, Any]:
    """Extract canonical column names and model fields from TFT config.

    Args:
        config_path: Path to tft_model.yaml.

    Returns:
        Dict with keys: timestamp_col, test_start, group_col, time_idx_col, horizon.

    Raises:
        ValueError: If config structure is invalid.
    """
    cfg = get_tft_cfg(config_path=config_path)
    splits = cfg.get("splits", {}) or {}
    model = cfg.get("model", {}) or {}
    if not isinstance(splits, dict) or not isinstance(model, dict):
        raise ValueError(f"Invalid TFT config structure: {config_path}")

    timestamp_col = splits.get("timestamp_col", "timestamp")
    test_start = splits.get("test_start")
    group_ids = model.get("group_ids", ["location"])
    time_idx = model.get("time_idx", "time_idx")
    horizon = model.get("max_prediction_length", 96)

    if not isinstance(timestamp_col, str) or not timestamp_col:
        timestamp_col = "timestamp"
    if not isinstance(group_ids, list) or not group_ids or not isinstance(group_ids[0], str):
        group_ids = ["location"]
    if not isinstance(time_idx, str) or not time_idx:
        time_idx = "time_idx"

    horizon = int(horizon)

    return {
        "timestamp_col": timestamp_col,
        "test_start": pd.Timestamp(test_start) if test_start is not None else None,
        "group_col": group_ids[0],
        "time_idx_col": time_idx,
        "horizon": horizon,
    }


def model_used_features(model: TemporalFusionTransformer) -> set[str]:
    """Extract full set of input features required by the model.

    Args:
        model: Loaded TemporalFusionTransformer.

    Returns:
        Set of feature name strings.
    """
    used: set[str] = set()
    dp = getattr(model.hparams, "dataset_parameters", {}) or {}
    for k in (
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "time_varying_known_categoricals",
        "time_varying_unknown_categoricals",
    ):
        for v in dp.get(k, []) or []:
            if isinstance(v, str):
                used.add(v)
    return used


def parse_predict_output(result: object) -> tuple[torch.Tensor | None, pd.DataFrame | None]:
    """Unpack varied TFT predict() outputs into a tensor and index DataFrame.

    Args:
        result: Raw output from model.predict().

    Returns:
        Tuple of (predictions tensor, index DataFrame).

    Raises:
        RuntimeError: If result format is not recognised.
    """
    if torch.is_tensor(result):
        return result, None  # type: ignore[return-value]

    if isinstance(result, (list, tuple)):
        preds_tensor = None
        index_df = None
        for item in result:
            if preds_tensor is None and torch.is_tensor(item):
                preds_tensor = item
            elif index_df is None and isinstance(item, pd.DataFrame):
                index_df = item
        return preds_tensor, index_df

    if isinstance(result, dict):
        preds = result.get("prediction")
        if preds is None:
            preds = result.get("predictions")
        return preds, result.get("index")

    raise RuntimeError(f"Unsupported predict output type: {type(result)}")


def fit_calibration(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    """Fit a linear calibration y_true ≈ a * y_pred + b via least-squares.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Tuple of (slope, intercept).
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    mask = np.isfinite(yt) & np.isfinite(yp)

    if mask.sum() < 10:
        return 1.0, 0.0

    yt, yp = yt[mask], yp[mask]

    # Trim 1st/99th percentile outliers to improve calibration robustness
    q1, q99 = np.percentile(yt, [1, 99])
    keep = (yt >= q1) & (yt <= q99)
    yt, yp = yt[keep], yp[keep]

    # Ordinary least-squares: [yp | 1] * [a, b]^T = yt
    a, b = np.linalg.lstsq(np.vstack([yp, np.ones_like(yp)]).T, yt, rcond=None)[0]
    return (float(a), float(b)) if np.isfinite(a) else (1.0, 0.0)


@torch.inference_mode()
def predict_timeseries(
    cfg: DictConfig,  # noqa: ARG001
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
    batch_size: int,
    tft_config_path: Path = DEFAULT_TFT_CONFIG_PATH,
) -> pd.DataFrame:
    """Generate power predictions for df using the TFT.

    Args:
        cfg: Full Hydra config.
        model: Loaded TemporalFusionTransformer.
        df: Input DataFrame with encoder + horizon window.
        batch_size: Dataloader batch size.
        tft_config_path: Path to tft_model.yaml.

    Returns:
        DataFrame with columns (group_col, timestamp_col, horizon_step, y_hat).

    Raises:
        KeyError: If required columns are missing from df.
        RuntimeError: If model or encoder is misconfigured.
    """
    tft = get_tft_fields(config_path=tft_config_path)
    ts_col = tft["timestamp_col"]
    test_start = tft["test_start"]
    group_col = tft["group_col"]
    time_idx_col = tft["time_idx_col"]
    horizon = int(tft["horizon"])

    # Basic input validation
    missing = [c for c in [ts_col, group_col, time_idx_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    # Extract site vocabulary from the trained model's categorical encoder
    ds_params = dict(model.hparams.dataset_parameters)
    loc_enc = ds_params.get("categorical_encoders", {}).get(group_col)

    if loc_enc is None:
        raise RuntimeError(f"Could not find '{group_col}' encoder in model checkpoint.")

    trained_sites = list(getattr(loc_enc, "classes_", []) or [])
    if not trained_sites:
        raise RuntimeError(f"Location encoder for '{group_col}' has empty classes_.")

    # Proxy unseen sites to avoid encoder index errors
    df = df.copy()
    df[group_col] = df[group_col].astype(str)
    unknown_mask = ~df[group_col].isin(trained_sites)
    if unknown_mask.any():
        proxy = str(trained_sites[0])
        logger.warning(
            "Proxying %d unknown sites to %s",
            int(df.loc[unknown_mask, group_col].nunique()),
            proxy,
        )
        df.loc[unknown_mask, group_col] = proxy

    # Decide prediction start index
    # Prefer config test_start; fallback to earliest timestamp if not set or no rows >= test_start
    if test_start is not None and (df[ts_col] >= test_start).any():
        pred_min_idx = int(df.loc[df[ts_col] >= test_start, time_idx_col].min())
    else:
        pred_min_idx = int(df[time_idx_col].min())
        if test_start is not None:
            logger.warning(
                "No rows found with %s >= %s; falling back to min(%s)=%d",
                ts_col,
                test_start,
                time_idx_col,
                pred_min_idx,
            )

    ds_params.update({"min_prediction_idx": pred_min_idx, "allow_missing_timesteps": True})

    pred_ds = TimeSeriesDataSet.from_parameters(ds_params, df, stop_randomization=True)
    loader = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    res = model.predict(
        loader, mode="prediction", return_index=True, trainer_kwargs={"logger": False}
    )
    preds, index = parse_predict_output(res)
    if preds is None or index is None:
        raise RuntimeError("Model predict() did not return both predictions and index.")

    # Select median quantile if probabilistic output shape is (B, H, Q)
    if preds.ndim == 3:
        preds = preds[..., preds.shape[-1] // 2]
    preds_np = preds.detach().cpu().numpy()

    # Map predictions back to original timestamps via (location, time_idx) lookup
    key_map = df.set_index([group_col, time_idx_col])[ts_col].to_dict()

    rows: list[dict[str, Any]] = []
    for i in range(len(index)):
        loc = str(index.iloc[i][group_col])
        t0 = int(index.iloc[i][time_idx_col])
        for h in range(horizon):
            ts = key_map.get((loc, t0 + h))
            if ts is not None and pd.notna(ts):
                rows.append({
                    group_col: loc,
                    ts_col: ts,
                    "horizon_step": h + 1,
                    "y_hat": float(preds_np[i, h]),
                })

    return pd.DataFrame(rows).sort_values([group_col, ts_col]).reset_index(drop=True)
