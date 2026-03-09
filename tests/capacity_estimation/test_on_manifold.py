"""Tests for capacity_estimation.on_manifold model_utils and data_utils."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.capacity_estimation.on_manifold.data_utils import (
    apply_daylight_constants,
    infer_orientation,
    pool_min_max,
    solar_score,
)
from src.capacity_estimation.on_manifold.model_utils import (
    fit_calibration,
    model_used_features,
    parse_predict_output,
)


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "splits": {"timestamp_col": "timestamp"},
        "model": {"target": "power"},
        "solar_weights": {"ssrd_w_m2": 1.0, "tcc": -1.0},
        "solar_scoring": {"pool_percentiles": {"low": 20, "high": 80}},
        "daylight": {"hours": [6, 19]},
        "orientation_inference": {
            "primary_solar_features": ["ssrd_w_m2"],
            "summer_months": [5, 9],
            "midday_hours": [11, 14],
            "min_samples": 5,
            "fallback_orientation": 1,
        },
    })


@pytest.fixture
def loc_df():
    n = 48
    timestamps = pd.date_range("2021-06-01 10:00", periods=n, freq="30min")
    rng = np.random.default_rng(42)
    irr = rng.uniform(100, 800, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "ssrd_w_m2": irr,
        "power": irr * 0.5 + rng.normal(0, 5, n),
    })


def test_fit_calibration_recovers_known_slope() -> None:
    y_pred = list(np.linspace(0, 10, 100))
    y_true = [2.0 * y + 1.0 for y in y_pred]
    a, b = fit_calibration(y_true, y_pred)
    assert abs(a - 2.0) < 0.01
    assert abs(b - 1.0) < 0.01


def test_fit_calibration_fallback_too_few_samples() -> None:
    a, b = fit_calibration([1.0, 2.0], [1.0, 2.0])
    assert (a, b) == (1.0, 0.0)


def test_fit_calibration_fallback_all_nan() -> None:
    a, b = fit_calibration([float("nan")] * 20, [float("nan")] * 20)
    assert (a, b) == (1.0, 0.0)


def test_solar_score_weighted_sum() -> None:
    vec = {"ssrd_w_m2": 10.0, "tcc": 5.0}
    weights = {"ssrd_w_m2": 1.0, "tcc": -1.0}
    assert solar_score(vec, weights) == pytest.approx(5.0)


def test_solar_score_ignores_non_finite() -> None:
    vec = {"ssrd_w_m2": float("nan"), "tcc": 5.0}
    weights = {"ssrd_w_m2": 1.0, "tcc": -1.0}
    assert solar_score(vec, weights) == pytest.approx(-5.0)


def test_pool_min_max_partitions_correctly(cfg) -> None:
    candidates = [{"ssrd_w_m2": float(i), "tcc": 0.0} for i in range(20)]
    low_pool, high_pool = pool_min_max(candidates, cfg)
    low_scores = [c["ssrd_w_m2"] for c in low_pool]
    high_scores = [c["ssrd_w_m2"] for c in high_pool]
    assert max(low_scores) < min(high_scores)


def test_pool_min_max_empty_input(cfg) -> None:
    assert pool_min_max([], cfg) == ([], [])


def test_pool_min_max_fallback_to_extremes(cfg) -> None:
    candidates = [{"ssrd_w_m2": 1.0, "tcc": 0.0}] * 5
    low_pool, high_pool = pool_min_max(candidates, cfg)
    assert len(low_pool) >= 1
    assert len(high_pool) >= 1


def test_infer_orientation_positive_correlation(cfg, loc_df) -> None:
    assert infer_orientation(loc_df, cfg) == 1


def test_infer_orientation_negative_correlation(cfg, loc_df) -> None:
    loc_df = loc_df.copy()
    loc_df["power"] = -loc_df["ssrd_w_m2"] + 1000.0
    assert infer_orientation(loc_df, cfg) == -1


def test_infer_orientation_fallback_missing_column(cfg) -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2021-06-01", periods=10, freq="h")})
    assert infer_orientation(df, cfg) == 1


def test_infer_orientation_fallback_too_few_samples(cfg, loc_df) -> None:
    loc_df = loc_df.copy()
    loc_df["timestamp"] = pd.date_range("2021-01-01 11:00", periods=len(loc_df), freq="30min")
    assert infer_orientation(loc_df, cfg) == 1


def test_parse_predict_output_tensor_only() -> None:
    t = torch.zeros(4, 8)
    preds, index = parse_predict_output(t)
    assert torch.equal(preds, t)
    assert index is None


def test_parse_predict_output_tuple() -> None:
    t = torch.zeros(4, 8)
    df = pd.DataFrame({"a": [1, 2]})
    preds, index = parse_predict_output((t, df))
    assert torch.equal(preds, t)
    assert isinstance(index, pd.DataFrame)


def test_parse_predict_output_dict() -> None:
    t = torch.zeros(4, 8)
    df = pd.DataFrame({"a": [1, 2]})
    preds, index = parse_predict_output({"prediction": t, "index": df})
    assert torch.equal(preds, t)
    assert isinstance(index, pd.DataFrame)


def test_parse_predict_output_raises_on_unknown() -> None:
    with pytest.raises(RuntimeError, match="Unsupported"):
        parse_predict_output("not_a_valid_output")


def test_apply_daylight_constants_modifies_within_hours(cfg) -> None:
    timestamps = pd.date_range("2021-06-01 06:00", periods=24, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "ssrd_w_m2": 0.0})
    out = apply_daylight_constants(df, {"ssrd_w_m2": 500.0}, cfg)
    daylight = out[out["timestamp"].dt.hour.between(6, 19)]
    night = out[~out["timestamp"].dt.hour.between(6, 19)]
    assert daylight["ssrd_w_m2"].to_numpy() == pytest.approx(500.0)
    assert night["ssrd_w_m2"].to_numpy() == pytest.approx(0.0)


def test_apply_daylight_constants_updates_sibling_features(cfg) -> None:
    timestamps = pd.date_range("2021-06-01 10:00", periods=4, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "ssrd_w_m2": 0.0,
        "ssrd_w_m2_lag_2h": 0.0,
        "ssrd_w_m2_roll_6h": 0.0,
    })
    out = apply_daylight_constants(df, {"ssrd_w_m2": 300.0}, cfg)
    assert out["ssrd_w_m2_lag_2h"].to_numpy() == pytest.approx(300.0)
    assert out["ssrd_w_m2_roll_6h"].to_numpy() == pytest.approx(300.0)


def test_apply_daylight_constants_noop_empty_mods(cfg) -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-06-01", periods=4, freq="h"),
        "ssrd_w_m2": 1.0,
    })
    out = apply_daylight_constants(df, {}, cfg)
    pd.testing.assert_frame_equal(df, out)


def test_model_used_features_returns_correct_set() -> None:
    model = MagicMock()
    model.hparams.dataset_parameters = {
        "time_varying_known_reals": ["ssrd_w_m2", "tcc"],
        "time_varying_unknown_reals": ["power"],
        "static_reals": [],
    }
    result = model_used_features(model)
    assert result == {"ssrd_w_m2", "tcc", "power"}


def test_model_used_features_ignores_non_strings() -> None:
    model = MagicMock()
    model.hparams.dataset_parameters = {
        "time_varying_known_reals": ["ssrd_w_m2", 42, None],
    }
    result = model_used_features(model)
    assert result == {"ssrd_w_m2"}
