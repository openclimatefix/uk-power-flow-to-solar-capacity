"""Shared fixtures for TFT tests."""

from __future__ import annotations

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.tft.model import TFTWithGRU, create_final_model


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return OmegaConf.create({
        "model": {
            "hidden_size": 16,
            "attention_head_size": 4,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "lstm_layers": 1,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "reduce_on_plateau_patience": 2,
            "head_diversity_lambda": 0.1,
            "max_epochs": 10,
            "scheduler": "cosine",
        },
    })


@pytest.fixture(scope="module")
def dataset() -> TimeSeriesDataSet:
    n = 12
    df = pd.DataFrame({
        "time_idx": list(range(n)) * 2,
        "location": ["a"] * n + ["b"] * n,
        "target": [float(i % 5) * 0.1 for i in range(n * 2)],
        "feat": [1.0] * n * 2,
    })
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["location"],
        max_encoder_length=4,
        max_prediction_length=2,
        time_varying_known_reals=["feat"],
        time_varying_unknown_reals=[],
        static_categoricals=[],
        static_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(
            groups=["location"], transformation="softplus"
        ),
    )


@pytest.fixture(scope="module")
def model(cfg: DictConfig, dataset: TimeSeriesDataSet) -> TFTWithGRU:
    return create_final_model(cfg, dataset)
