"""Tests for capacity_estimation.historical_analogue."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.capacity_estimation.historical_analogue import (
    ensure_time_idx,
    estimate_capacity_for_window,
    estimate_site_capacity_year,
)

_ANALOGUE_PATCH = "src.capacity_estimation.historical_analogue.estimate_capacity_for_window"


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "peak_hours": [11, 15],
        "analogue_percentile_high": 95,
        "analogue_percentile_low": 5,
        "min_samples_per_site": 10,
        "min_windows_for_estimate": 2,
        "step_size": 50,
    })


def _make_df(n: int, year: int = 2022, freq: str = "30min") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    timestamps = pd.date_range(f"{year}-06-01", periods=n, freq=freq)
    return pd.DataFrame({
        "timestamp": timestamps,
        "location": "site_a",
        "ssrd_w_m2": rng.uniform(0, 800, n),
        "t2m_c": rng.uniform(5, 25, n),
        "time_idx": np.arange(n),
    })


def _make_model(max_enc: int = 10, max_pred: int = 5) -> MagicMock:
    model = MagicMock()
    model.dataset_parameters = {
        "max_encoder_length": max_enc,
        "max_prediction_length": max_pred,
    }
    return model


def test_ensure_time_idx_removes_timezone() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=4, freq="30min", tz="UTC")})
    out = ensure_time_idx(df)
    assert out["timestamp"].dt.tz is None


def test_ensure_time_idx_starts_at_zero() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=4, freq="30min", tz="UTC")})
    out = ensure_time_idx(df)
    assert out["time_idx"].iloc[0] == 0


def test_ensure_time_idx_increments_by_one_per_half_hour() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=6, freq="30min", tz="UTC")})
    out = ensure_time_idx(df)
    assert list(out["time_idx"]) == [0, 1, 2, 3, 4, 5]


def test_ensure_time_idx_does_not_mutate_input() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=4, freq="30min", tz="UTC")})
    original_ts = df["timestamp"].copy()
    ensure_time_idx(df)
    pd.testing.assert_series_equal(df["timestamp"], original_ts)


def test_ensure_time_idx_handles_naive_timestamps() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=4, freq="30min")})
    out = ensure_time_idx(df)
    assert "time_idx" in out.columns


def test_estimate_capacity_returns_none_if_window_too_short(cfg) -> None:
    model = _make_model(max_enc=10, max_pred=5)
    df = _make_df(10)
    result = estimate_capacity_for_window(df, df, model, ["ssrd_w_m2"], [2022], cfg)
    assert result is None


def test_estimate_capacity_returns_none_if_no_analogue_years(cfg) -> None:
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(20, year=2022)
    result = estimate_capacity_for_window(df, df, model, ["ssrd_w_m2"], [1990], cfg)
    assert result is None


def test_estimate_capacity_returns_none_if_no_peak_hours_in_window(cfg) -> None:
    cfg = OmegaConf.create({**OmegaConf.to_container(cfg), "peak_hours": [23, 23]})
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(20, year=2022)

    with patch("src.capacity_estimation.historical_analogue.TimeSeriesDataSet") as mock_ds:
        mock_dl = MagicMock()
        mock_ds.from_parameters.return_value.to_dataloader.return_value = mock_dl
        tensor_mock = MagicMock()
        tensor_mock.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = (
            np.ones(3)
        )
        model.predict.return_value = tensor_mock
        result = estimate_capacity_for_window(df, df, model, ["ssrd_w_m2"], [2022], cfg)

    assert result is None


def test_estimate_capacity_non_negative(cfg) -> None:
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(50, year=2022)

    with patch("src.capacity_estimation.historical_analogue.TimeSeriesDataSet") as mock_ds:
        mock_dl = MagicMock()
        mock_ds.from_parameters.return_value.to_dataloader.return_value = mock_dl
        tensor_mock = MagicMock()
        tensor_mock.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = (
            np.array([10.0, 20.0, 30.0])
        )
        model.predict.return_value = tensor_mock

        result = estimate_capacity_for_window(df, df, model, ["ssrd_w_m2"], [2022], cfg)

    if result is not None:
        assert result >= 0.0


def test_estimate_site_capacity_returns_none_too_few_rows(cfg) -> None:
    model = _make_model()
    df = _make_df(5, year=2022)
    result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)
    assert result is None


def test_estimate_site_capacity_returns_none_wrong_year(cfg) -> None:
    model = _make_model()
    df = _make_df(500, year=2022)
    result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2099, [2022], cfg)
    assert result is None


def test_estimate_site_capacity_returns_none_too_few_windows(cfg) -> None:
    cfg = OmegaConf.create({**OmegaConf.to_container(cfg), "min_windows_for_estimate": 9999})
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(500, year=2022)

    with patch(
        _ANALOGUE_PATCH,
        return_value=1.0,
    ):
        result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)

    assert result is None


def test_estimate_site_capacity_returns_dict_with_correct_keys(cfg) -> None:
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(500, year=2022)

    with patch(
        _ANALOGUE_PATCH,
        return_value=2.5,
    ):
        result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)

    assert result is not None
    assert set(result.keys()) == {"mean", "p90", "n_samples"}


def test_estimate_site_capacity_mean_non_negative(cfg) -> None:
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(500, year=2022)

    with patch(
        _ANALOGUE_PATCH,
        return_value=3.0,
    ):
        result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)

    assert result["mean"] >= 0.0


def test_estimate_site_capacity_p90_gte_mean(cfg) -> None:
    from itertools import cycle

    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(500, year=2022)

    with patch(
        _ANALOGUE_PATCH,
        side_effect=cycle([1.0, 2.0, 3.0, 4.0, 5.0]),
    ):
        result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)

    assert result["p90"] >= result["mean"]


def test_estimate_site_capacity_n_samples_positive(cfg) -> None:
    model = _make_model(max_enc=5, max_pred=3)
    df = _make_df(500, year=2022)

    with patch(
        _ANALOGUE_PATCH,
        return_value=1.0,
    ):
        result = estimate_site_capacity_year(df, df, model, ["ssrd_w_m2"], 2022, [2022], cfg)

    assert result["n_samples"] > 0
