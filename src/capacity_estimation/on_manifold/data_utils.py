"""Data utility functions for capacity estimation scenarios."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Cache for configurations to avoid redundant I/O
_SOLAR_CFG_CACHE: DictConfig | None = None
_TFT_CFG_CACHE: DictConfig | None = None


def solar_score(
    vec: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Calculates a scalar score representing solar potential for a weather vector.

    Args:
        vec: Weather feature vector.
        weights: Feature weights from configuration.

    Returns:
        Weighted scalar score.
    """
    s = 0.0
    for k, v in vec.items():
        w = weights.get(k)
        if w is not None and np.isfinite(v):
            s += w * float(v)
    return float(s)


def pool_min_max(
    candidates: list[dict[str, float]],
    cfg: DictConfig,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Partitions candidate vectors into extreme 'low' and 'high' solar pools.

    Args:
        candidates: List of generated weather vectors.
        cfg: Hydra config containing solar_weights and solar_scoring.

    Returns:
        Tuple of (low_solar_pool, high_solar_pool).
    """
    if not candidates:
        return [], []

    weights = cfg.get("solar_weights", {})
    scoring = cfg.get("solar_scoring", {})
    pool_cfg = scoring.get("pool_percentiles", {"low": 20, "high": 80})

    scores = np.array([solar_score(c, weights) for c in candidates], float)

    lo_thr = np.nanpercentile(scores, int(pool_cfg["low"]))
    hi_thr = np.nanpercentile(scores, int(pool_cfg["high"]))

    low_pool = [c for c, s in zip(
        candidates, scores, strict=True
    ) if np.isfinite(s) and s <= lo_thr]

    high_pool = [c for c, s in zip(
        candidates, scores, strict=True
    ) if np.isfinite(s) and s >= hi_thr]

    # Fallback to absolute extremes if percentiles yield empty lists
    if not low_pool:
        low_pool = [candidates[int(np.nanargmin(scores))]]
    if not high_pool:
        high_pool = [candidates[int(np.nanargmax(scores))]]

    return low_pool, high_pool


def infer_orientation(
    loc_df: pd.DataFrame,
    cfg: DictConfig,
) -> int:
    """Detects if site power output correlates positively or negatively with solar input.

    Args:
        loc_df: DataFrame for a specific location.
        cfg: Hydra config containing orientation_inference, splits, and model.

    Returns:
        1 for positive correlation, -1 for negative.
    """
    pol = cfg.get("orientation_inference", {})
    ts_col = cfg.splits.timestamp_col
    p_col = cfg.model.target

    if p_col not in loc_df.columns or ts_col not in loc_df.columns:
        return int(pol.get("fallback_orientation", 1))

    primary_features = [c for c in pol.get("primary_solar_features", []) if c in loc_df.columns]
    if not primary_features:
        return int(pol.get("fallback_orientation", 1))

    sm = pol.get("summer_months", [5, 9])
    mh = pol.get("midday_hours", [11, 14])

    # Filter for high-irradiance context to check correlation
    mask = (
        loc_df[ts_col].dt.month.between(int(sm[0]), int(sm[1])) &
        loc_df[ts_col].dt.hour.between(int(mh[0]), int(mh[1]))
    )
    data = loc_df.loc[mask, [primary_features[0], p_col]].dropna()

    if len(data) < int(pol.get("min_samples", 20)):
        return int(pol.get("fallback_orientation", 1))

    corr = data[primary_features[0]].corr(data[p_col])
    return -1 if corr < 0 else 1


def apply_daylight_constants(
    df: pd.DataFrame,
    mods: dict[str, float],
    cfg: DictConfig,
) -> pd.DataFrame:
    """Overwrites weather features with scenario values during specific hours.

    Args:
        df: Input location DataFrame.
        mods: Weather modifications to apply.
        cfg: Hydra config containing daylight hours and splits.

    Returns:
        DataFrame with scenario modifications applied.
    """
    if not mods:
        return df

    ts_col = cfg.splits.timestamp_col
    daylight = cfg.get("daylight", {})
    hours = daylight.get("hours", [6, 19])

    out = df.copy()
    start_h, end_h = int(hours[0]), int(hours[1])
    mask = out[ts_col].dt.hour.between(start_h, end_h)

    for k, v in mods.items():
        if k in out.columns:
            out.loc[mask, k] = v

            # Maintain consistency for engineered 'sibling' features (lags, rolling)
            sibs = [
                c for c in out.columns
                if c != k and c.startswith(k) and any(x in c for x in ["lag", "roll", "diff"])
            ]
            for s in sibs:
                out.loc[mask, s] = v

    return out


def ensure_sorted_and_time_idx(
    df: pd.DataFrame,
    time_idx_col: str,
    group_col: str,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Ensures data is chronologically ordered with a continuous time index per site.

    Args:
        df: Input DataFrame.
        time_idx_col: Target column name for the integer index.
        group_col: Column name identifying sites.
        cfg: Hydra config containing timestamp_col.

    Returns:
        Sorted DataFrame with continuous time index.
    """
    ts_col = cfg.splits.timestamp_col
    if ts_col not in df.columns:
        raise KeyError(f"Missing timestamp column '{ts_col}' in df")

    df = df.sort_values([group_col, ts_col]).reset_index(drop=True)
    df[time_idx_col] = df.groupby(group_col, sort=False).cumcount().astype("int64")
    return df
