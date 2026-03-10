from __future__ import annotations

import inspect

import pandas as pd

from src.nhits.data import _load_parquet_columns, create_production_datasets
from src.nhits.model import create_nhits_model
from src.nhits.utils import get_nhits_resume_checkpoint


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


def test_create_production_datasets_signature() -> None:
    sig = inspect.signature(create_production_datasets)
    for param in ["cfg", "dataset_path"]:
        assert param in sig.parameters


def test_create_nhits_model_signature() -> None:
    sig = inspect.signature(create_nhits_model)
    for param in ["cfg", "training_dataset"]:
        assert param in sig.parameters


def test_get_nhits_resume_checkpoint_returns_none_for_missing(tmp_path) -> None:
    result = get_nhits_resume_checkpoint(tmp_path / "nonexistent")
    assert result is None


def test_get_nhits_resume_checkpoint_finds_last_ckpt(tmp_path) -> None:
    ckpt_dir = tmp_path / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    last = ckpt_dir / "last.ckpt"
    last.write_text("fake")
    result = get_nhits_resume_checkpoint(tmp_path)
    assert result == str(last)
