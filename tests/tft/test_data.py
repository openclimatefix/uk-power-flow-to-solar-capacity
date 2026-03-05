"""Tests for src.tft.data."""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.tft.data import create_production_datasets
from src.tft.utils import intersect_features


@pytest.fixture
def cfg() -> DictConfig:
    return OmegaConf.create({
        "model": {
            "group_ids": ["location"],
            "target": "active_power_mw_clean",
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_reals": ["u10", "v10"],
            "time_varying_unknown_reals": [],
            "max_encoder_length": 4,
            "max_prediction_length": 2,
            "add_target_scales": True,
        },
        "train_split": 0.6,
        "val_split": 0.9,
    })


@pytest.fixture
def parquet_path(tmp_path) -> str:
    n = 20
    pdf = pd.DataFrame({
        "location": ["site_1"] * n,
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="30min"),
        "active_power_mw_clean": [float(i) * 0.1 for i in range(n)],
        "u10": [1.0] * n,
        "v10": [0.5] * n,
    })
    path = str(tmp_path / "data.parquet")
    pdf.to_parquet(path, index=False)
    return path


@pytest.fixture
def time_cfg(cfg: DictConfig) -> DictConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw["splits"] = {
        "strategy": "by_time",
        "timestamp_col": "timestamp",
        "train_end": "2021-01-01 05:00:00",
        "val_start": "2021-01-01 06:00:00",
        "val_end": "2021-01-01 08:00:00",
    }
    return OmegaConf.create(raw)


def test_intersect_features_returns_present_cols() -> None:
    result = intersect_features(["a", "b", "c"], ["a", "c", "x"])
    assert result == ["a", "c"]


def test_intersect_features_warns_on_missing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        intersect_features(["a", "b"], ["a", "missing"])
    assert "Missing" in caplog.text


def test_intersect_features_empty_keys() -> None:
    assert intersect_features(["a", "b"], []) == []


def test_fractional_split_returns_three_tuple(cfg: DictConfig, parquet_path: str) -> None:
    result = create_production_datasets(cfg, parquet_path)
    assert len(result) == 3


def test_fractional_split_time_idx_present(cfg: DictConfig, parquet_path: str) -> None:
    _, _, pdf = create_production_datasets(cfg, parquet_path)
    assert "time_idx" in pdf.columns


def test_fractional_split_datasets_nonempty(cfg: DictConfig, parquet_path: str) -> None:
    training_ds, val_ds, _ = create_production_datasets(cfg, parquet_path)
    assert len(training_ds) > 0
    assert len(val_ds) > 0


def test_fractional_split_encoder_length(cfg: DictConfig, parquet_path: str) -> None:
    training_ds, _, _ = create_production_datasets(cfg, parquet_path)
    assert training_ds.max_encoder_length == cfg.model.max_encoder_length


def test_by_time_split_datasets_nonempty(time_cfg: DictConfig, parquet_path: str) -> None:
    training_ds, val_ds, _ = create_production_datasets(time_cfg, parquet_path)
    assert len(training_ds) > 0
    assert len(val_ds) > 0


def test_by_time_split_val_after_train(time_cfg: DictConfig, parquet_path: str) -> None:
    create_production_datasets(time_cfg, parquet_path)
    train_end = pd.Timestamp(time_cfg.splits.train_end)
    val_start = pd.Timestamp(time_cfg.splits.val_start)
    assert train_end < val_start
