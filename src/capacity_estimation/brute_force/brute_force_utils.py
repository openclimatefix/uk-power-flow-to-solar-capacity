"""
Utility functions for brute-force scenario simulation.

The brute force method pushes important solar/weather features
toward high and low extremes using site-specific
quantiles and clipping to empirical bounds.
"""

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
) -> tuple[dict, dict, dict]:
    """Compute per location quantiles and bounds."""
    data = df[cols].dropna(how="all")
    if data.empty:
        return {}, {}, {}

    q = data.quantile([lo_q, hi_q])
    lo = q.iloc[0].to_dict()
    hi = q.iloc[1].to_dict()

    dmin = data.min().to_dict()
    dmax = data.max().to_dict()

    return lo, hi, (dmin, dmax)


def _clip_push(val: float, factor: float, vmin: float, vmax: float) -> float:
    """Push value and clip to bounds."""
    return float(np.clip(val * factor, vmin, vmax))


def build_coherent_mods(
    df: pd.DataFrame,
    cols: list[str],
    hi_q: float = 0.9,
    lo_q: float = 0.1,
    push_hi: float = 1.1,
    push_lo: float = 0.9,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Create scenario modifications.

    Scenario A: high solar
    Scenario B: low solar
    """

    lo, hi, (dmin, dmax) = _site_quantiles(df, cols, lo_q, hi_q)

    mins = {}
    maxs = {}

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
            maxs[c] = _clip_push(lo_v, push_lo, vmin, vmax)
            mins[c] = _clip_push(hi_v, push_hi, vmin, vmax)

        elif is_elev or is_irr:
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
    """Apply feature modifications during daylight."""
    if not mods:
        return df

    out = df.copy()

    start_h, end_h = daylight_hours
    mask = out["timestamp"].dt.hour.between(start_h, end_h)

    for k, v in mods.items():
        if k in out.columns:
            out.loc[mask, k] = v

    return out
