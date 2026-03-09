from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.tft.infer import build_pred_dataset, chunked_predict_and_write, run_predict_one_chunk


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {
            "group_ids": ["location"],
            "time_idx": "time_idx",
            "max_encoder_length": 4,
            "max_prediction_length": 2,
        },
        "splits": {
            "test_start": "2021-01-01 02:00:00",
            "test_end": "2021-01-01 04:00:00",
        },
        "paths": {
            "dataset_path": "dummy.parquet",
        },
    })


@pytest.fixture
def sample_df():
    n = 20
    return pd.DataFrame({
        "location": ["site_1"] * n,
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="30min"),
        "time_idx": list(range(n)),
        "target": [float(i) * 0.1 for i in range(n)],
    })


def test_build_pred_dataset_returns_none_when_no_test_rows(cfg, sample_df) -> None:
    cfg_early = OmegaConf.merge(
        cfg, {"splits": {"test_start": "2025-01-01", "test_end": "2025-01-02"}}
    )
    model = MagicMock()
    model.hparams.dataset_parameters = {}
    result, idx = build_pred_dataset(cfg_early, sample_df, model)
    assert result is None
    assert idx == -1


def test_run_predict_one_chunk_returns_empty_when_pred_ds_is_none(cfg, sample_df) -> None:
    model = MagicMock()
    with patch("src.tft.infer.build_pred_dataset", return_value=(None, -1)):
        result = run_predict_one_chunk(cfg, sample_df, model, batch_size=4)
    assert result.empty


def test_run_predict_one_chunk_returns_empty_when_no_pred_ds(cfg, sample_df) -> None:
    model = MagicMock()
    with patch("src.tft.infer.build_pred_dataset", return_value=(None, -1)):
        result = run_predict_one_chunk(cfg, sample_df, model, batch_size=4)
    assert result.empty


def test_chunked_predict_and_write_skips_empty_chunks(cfg, tmp_path) -> None:
    model = MagicMock()
    out_path = tmp_path / "out.parquet"

    with (
        patch("src.tft.infer.read_test_slice", return_value=pd.DataFrame()),
        patch("src.tft.infer.run_predict_one_chunk", return_value=pd.DataFrame()),
    ):
        chunked_predict_and_write(cfg, ["site_1"], 1, model, out_path, batch_size=4)

    assert not out_path.exists()


def test_chunked_predict_and_write_creates_parquet(cfg, tmp_path) -> None:
    model = MagicMock()
    out_path = tmp_path / "out.parquet"

    pred_df = pd.DataFrame({
        "location": ["site_1"],
        "timestamp": [pd.Timestamp("2021-01-01 02:00:00")],
        "horizon_step": [1],
        "y_hat": [0.5],
    })

    with (
        patch("src.tft.infer.read_test_slice", return_value=pred_df),
        patch("src.tft.infer.run_predict_one_chunk", return_value=pred_df),
    ):
        chunked_predict_and_write(cfg, ["site_1"], 1, model, out_path, batch_size=4)

    assert out_path.exists()
