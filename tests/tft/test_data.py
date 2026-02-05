import dask.dataframe as dd
import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from src.tft.data import create_production_datasets
from src.tft.utils import intersect_features


@pytest.fixture
def cfg() -> DictConfig:
    with initialize(version_base=None, config_path="../../configs/tft"):
        return compose(config_name="tft_model", overrides=["production_locations=null"])


def test_intersect_features_filters_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    existing_cols = ["a", "b", "c"]
    keys = ["a", "x", "c", "y"]

    with caplog.at_level("WARNING"):
        result = intersect_features(existing_cols, keys)

    assert result == ["a", "c"]
    assert "Missing" in caplog.text


def test_create_production_datasets_fractional_split(
    cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.model.max_encoder_length = 2
    cfg.model.max_prediction_length = 1
    cfg.splits.strategy = "fractional"
    cfg.train_split = 0.5
    cfg.val_split = 0.9

    data = {
        "location": ["site_1"] * 10,
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="30min"),
        "active_power_mw_clean": [0.1] * 10,
    }
    pdf = pd.DataFrame(data)

    for col in list(cfg.model.time_varying_known_reals) + list(cfg.model.static_reals):
        pdf[col] = 0.0

    ddf = dd.from_pandas(pdf, npartitions=1)
    # Fix ARG005: Use underscores for unused lambda arguments
    monkeypatch.setattr("src.tft.data.dd.read_parquet", lambda *_, **__: ddf)

    training_dataset, validation_dataset, backing_df = create_production_datasets(
        cfg=cfg,
        dataset_path="dummy/path.parquet",
        num_locations=None,
    )

    assert "time_idx" in backing_df.columns
    assert training_dataset.max_encoder_length == 2
    assert len(training_dataset) > 0
    assert len(validation_dataset) > 0


def test_create_production_datasets_by_time_split(
    cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.model.max_encoder_length = 2
    cfg.model.max_prediction_length = 1

    cfg.splits.strategy = "by_time"
    cfg.splits.train_end = "2021-01-01 05:00:00"
    cfg.splits.val_start = "2021-01-01 06:00:00"
    cfg.splits.val_end = "2021-01-01 08:00:00"

    data = {
        "location": ["site_1"] * 20,
        "timestamp": pd.date_range("2021-01-01", periods=20, freq="30min"),
        "active_power_mw_clean": [0.1] * 20,
    }
    pdf = pd.DataFrame(data)

    for col in list(cfg.model.time_varying_known_reals) + list(cfg.model.static_reals):
        pdf[col] = 0.0

    ddf = dd.from_pandas(pdf, npartitions=1)
    monkeypatch.setattr("src.tft.data.dd.read_parquet", lambda *_, **__: ddf)

    training_dataset, validation_dataset, _ = create_production_datasets(
        cfg=cfg,
        dataset_path="dummy/path.parquet"
    )

    assert len(training_dataset) > 0
    assert len(validation_dataset) > 0
