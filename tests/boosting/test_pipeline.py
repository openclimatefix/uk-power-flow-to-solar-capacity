from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.boosting.data import _load_parquet_columns, create_train_val_split
from src.boosting.model import create_xgboost_model
from src.boosting.utils import compute_metrics, save_xgboost_artifacts


def test_load_parquet_columns_returns_dataframe(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    result = _load_parquet_columns(str(p), ["a"])
    assert list(result.columns) == ["a"]
    assert len(result) == 2


def test_load_parquet_columns_ignores_missing_cols(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    result = _load_parquet_columns(str(p), ["a", "nonexistent"])
    assert "nonexistent" not in result.columns


def test_create_train_val_split_signature() -> None:
    sig = inspect.signature(create_train_val_split)
    for param in ["cfg", "dataset_path"]:
        assert param in sig.parameters


def test_create_xgboost_model_signature() -> None:
    assert "cfg" in inspect.signature(create_xgboost_model).parameters


def test_compute_metrics_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["smape"] == pytest.approx(0.0)


def test_compute_metrics_returns_expected_keys() -> None:
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y + 0.1)
    assert set(metrics.keys()) == {"mae", "rmse", "smape"}


def test_compute_metrics_non_negative() -> None:
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    metrics = compute_metrics(y_true, y_pred)
    for v in metrics.values():
        assert v >= 0.0


def test_save_xgboost_artifacts_creates_files(tmp_path) -> None:
    import xgboost as xgb
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "model": {"target": "y", "features": ["x"]},
        "training": {"learning_rate": 0.1},
        "paths": {"results_path": str(tmp_path)},
    })

    model = xgb.XGBRegressor(n_estimators=5)
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model.fit(x, y)

    save_xgboost_artifacts(
        model=model,
        cfg=cfg,
        metrics={"mae": 0.1, "rmse": 0.2, "smape": 1.0},
        feature_cols=["x"],
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "production_xgboost_model.json").exists()
    assert (tmp_path / "production_xgboost_metadata.json").exists()
