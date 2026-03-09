"""Tests for capacity_estimation.brute_force.brute_force_utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.capacity_estimation.brute_force.brute_force_utils import (
    _clip_push,
    _site_quantiles,
    apply_scenario,
    build_coherent_mods,
)


@pytest.fixture
def site_df() -> pd.DataFrame:
    n = 100
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2022-06-01 00:00", periods=n, freq="30min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "ssrd_w_m2": rng.uniform(0, 800, n),
        "tcc": rng.uniform(0, 1, n),
        "solar_elevation_angle": rng.uniform(0, 90, n),
        "solar_zenith_angle": rng.uniform(0, 90, n),
        "t2m_c": rng.uniform(5, 25, n),
    })


def test_clip_push_scales_value() -> None:
    assert _clip_push(10.0, 1.1, 0.0, 20.0) == pytest.approx(11.0)


def test_clip_push_clips_to_max() -> None:
    assert _clip_push(10.0, 2.0, 0.0, 15.0) == pytest.approx(15.0)


def test_clip_push_clips_to_min() -> None:
    assert _clip_push(1.0, 0.5, 2.0, 10.0) == pytest.approx(2.0)


def test_clip_push_zero_factor() -> None:
    assert _clip_push(5.0, 0.0, 0.0, 10.0) == pytest.approx(0.0)


def test_site_quantiles_returns_correct_shape(site_df) -> None:
    lo, hi, (dmin, dmax) = _site_quantiles(site_df, ["ssrd_w_m2", "tcc"], 0.1, 0.9)
    assert set(lo.keys()) == {"ssrd_w_m2", "tcc"}
    assert set(hi.keys()) == {"ssrd_w_m2", "tcc"}
    assert set(dmin.keys()) == {"ssrd_w_m2", "tcc"}
    assert set(dmax.keys()) == {"ssrd_w_m2", "tcc"}


def test_site_quantiles_lo_less_than_hi(site_df) -> None:
    lo, hi, _ = _site_quantiles(site_df, ["ssrd_w_m2"], 0.1, 0.9)
    assert lo["ssrd_w_m2"] < hi["ssrd_w_m2"]


def test_site_quantiles_empty_df_returns_empty_dicts() -> None:
    df = pd.DataFrame({"ssrd_w_m2": pd.Series([], dtype=float)})
    lo, hi, (dmin, dmax) = _site_quantiles(df, ["ssrd_w_m2"], 0.1, 0.9)
    assert lo == {}
    assert hi == {}
    assert dmin == {}
    assert dmax == {}


def test_site_quantiles_all_nan_returns_empty_dicts() -> None:
    df = pd.DataFrame({"ssrd_w_m2": [float("nan")] * 10})
    lo, _hi, (_dmin, _dmax) = _site_quantiles(df, ["ssrd_w_m2"], 0.1, 0.9)
    assert lo == {}


def test_build_coherent_mods_irradiance_high_exceeds_low(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["ssrd_w_m2"])
    assert maxs["ssrd_w_m2"] >= mins["ssrd_w_m2"]


def test_build_coherent_mods_cloud_low_exceeds_high(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["tcc"])
    assert maxs["tcc"] <= mins["tcc"]


def test_build_coherent_mods_zenith_inverted(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["solar_zenith_angle"])
    assert maxs["solar_zenith_angle"] <= mins["solar_zenith_angle"]


def test_build_coherent_mods_elevation_direct(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["solar_elevation_angle"])
    assert maxs["solar_elevation_angle"] >= mins["solar_elevation_angle"]


def test_build_coherent_mods_unknown_feature_uses_default_push(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["t2m_c"])
    assert maxs["t2m_c"] >= mins["t2m_c"]


def test_build_coherent_mods_respects_empirical_bounds(site_df) -> None:
    _mins, maxs = build_coherent_mods(site_df, ["ssrd_w_m2"], push_hi=10.0)
    assert maxs["ssrd_w_m2"] <= site_df["ssrd_w_m2"].max()


def test_build_coherent_mods_skips_missing_cols(site_df) -> None:
    mins, maxs = build_coherent_mods(site_df, ["nonexistent_col"])
    assert "nonexistent_col" not in mins
    assert "nonexistent_col" not in maxs


def test_apply_scenario_modifies_within_daylight(site_df) -> None:
    out = apply_scenario(site_df, {"ssrd_w_m2": 999.0}, daylight_hours=(6, 19))
    daylight = out[out["timestamp"].dt.hour.between(6, 19)]
    assert daylight["ssrd_w_m2"].to_numpy() == pytest.approx(999.0)


def test_apply_scenario_does_not_modify_night(site_df) -> None:
    original_night = site_df[~site_df["timestamp"].dt.hour.between(6, 19)]["ssrd_w_m2"].copy()
    out = apply_scenario(site_df, {"ssrd_w_m2": 999.0}, daylight_hours=(6, 19))
    night = out[~out["timestamp"].dt.hour.between(6, 19)]["ssrd_w_m2"]
    pd.testing.assert_series_equal(
        night.reset_index(drop=True),
        original_night.reset_index(drop=True),
    )


def test_apply_scenario_noop_on_empty_mods(site_df) -> None:
    out = apply_scenario(site_df, {})
    pd.testing.assert_frame_equal(site_df, out)


def test_apply_scenario_ignores_unknown_keys(site_df) -> None:
    out = apply_scenario(site_df, {"nonexistent_col": 1.0})
    pd.testing.assert_frame_equal(site_df, out)


def test_apply_scenario_does_not_mutate_input(site_df) -> None:
    original = site_df["ssrd_w_m2"].copy()
    apply_scenario(site_df, {"ssrd_w_m2": 999.0})
    pd.testing.assert_series_equal(site_df["ssrd_w_m2"], original)


def test_apply_scenario_custom_daylight_hours(site_df) -> None:
    out = apply_scenario(site_df, {"tcc": 0.0}, daylight_hours=(10, 14))
    in_window = out[out["timestamp"].dt.hour.between(10, 14)]
    outside = out[~out["timestamp"].dt.hour.between(10, 14)]
    assert in_window["tcc"].to_numpy() == pytest.approx(0.0)
    assert outside["tcc"].to_numpy() != pytest.approx([0.0] * len(outside))
