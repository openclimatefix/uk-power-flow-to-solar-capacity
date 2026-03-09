"""Utility functions for brute-force solar scenario simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

_IRRADIANCE = ["ghi", "dni", "dhi", "cs_ghi", "ssrd_w_m2", "ssr_w_m2"]
_CLOUD = ["tcc", "cloud", "cloud_cover", "total_cloud_cover"]
_ZENITH = ["solar_zenith_angle", "zenith_angle"]
_ELEV = ["solar_elevation_angle", "elevation_angle"]


def _site_quantiles(
    df: pd.DataFrame,
    cols: list[str],
    lo_q: float,
    hi_q: float,
) -> tuple[dict[str, float], dict[str, float], tuple[dict[str, float], dict[str, float]]]:
    """Compute per-feature quantiles and empirical bounds.

    Args:
        df: Input DataFrame for one site.
        cols: Feature columns to compute quantiles over.
        lo_q: Lower quantile fraction.
        hi_q: Upper quantile fraction.

    Returns:
        Tuple of (lo_quantiles, hi_quantiles, (col_mins, col_maxs)).
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {}, {}, ({}, {})
    data = df[cols].dropna(how="all")
    if data.empty:
        return {}, {}, ({}, {})

    q = data.quantile([lo_q, hi_q])
    lo = q.iloc[0].to_dict()
    hi = q.iloc[1].to_dict()
    return lo, hi, (data.min().to_dict(), data.max().to_dict())


def _clip_push(val: float, factor: float, vmin: float, vmax: float) -> float:
    """Scale val by factor and clip to [vmin, vmax].

    Args:
        val: Base value.
        factor: Multiplicative push factor.
        vmin: Minimum empirical bound.
        vmax: Maximum empirical bound.

    Returns:
        Clipped float.
    """
    return float(np.clip(val * factor, vmin, vmax))


def build_coherent_mods(
    df: pd.DataFrame,
    cols: list[str],
    hi_q: float = 0.9,
    lo_q: float = 0.1,
    push_hi: float = 1.1,
    push_lo: float = 0.9,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build high- and low-solar scenario modifications from site quantiles.

    Cloud and zenith features are inverted (low values → high solar).
    Irradiance and elevation features are pushed directly.

    Args:
        df: Single-site DataFrame.
        cols: Weather feature columns to modify.
        hi_q: Upper quantile for high-solar scenario.
        lo_q: Lower quantile for low-solar scenario.
        push_hi: Multiplicative factor for high-solar push.
        push_lo: Multiplicative factor for low-solar push.

    Returns:
        Tuple of (low_solar_mods, high_solar_mods).
    """
    lo, hi, (dmin, dmax) = _site_quantiles(df, cols, lo_q, hi_q)

    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}

    for c in cols:
        if c not in lo:
            continue

        lo_v = lo[c]
        hi_v = hi[c]
        vmin = dmin[c]
        vmax = dmax[c]

        is_cloud = any(k in c for k in _CLOUD)
        is_zenith = any(k in c for k in _ZENITH)
        is_elev = any(k in c for k in _ELEV)
        is_irr = any(k in c for k in _IRRADIANCE)

        if is_cloud or is_zenith:
            # Lower cloud/zenith → higher solar output
            maxs[c] = _clip_push(lo_v, push_lo, vmin, vmax)
            mins[c] = _clip_push(hi_v, push_hi, vmin, vmax)
        elif is_elev or is_irr:
            # Higher elevation/irradiance → higher solar output
            maxs[c] = _clip_push(hi_v, push_hi, vmin, vmax)
            mins[c] = _clip_push(lo_v, push_lo, vmin, vmax)
        else:
            maxs[c] = _clip_push(hi_v, push_hi, vmin, vmax)
            mins[c] = _clip_push(lo_v, push_lo, vmin, vmax)

    return mins, maxs


def apply_scenario(
    df: pd.DataFrame,
    mods: dict[str, float],
    daylight_hours: tuple[int, int] = (6, 19),
) -> pd.DataFrame:
    """Overwrite weather features with scenario values during daylight hours.

    Args:
        df: Input location DataFrame with a timestamp column.
        mods: Feature name to replacement value mapping.
        daylight_hours: Inclusive (start_hour, end_hour) for modifications.

    Returns:
        Copy of df with scenario modifications applied.
    """
    if not mods:
        return df

    out = df.copy()
    start_h, end_h = daylight_hours
    mask = out["timestamp"].dt.hour.between(start_h, end_h)

    for k, v in mods.items():
        if k in out.columns:
            out.loc[mask, k] = v

    return out
