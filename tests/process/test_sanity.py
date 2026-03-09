"""Tests for src.process.sanity."""

from __future__ import annotations

import inspect

from process.sanity import check_csv_basic, check_file_exists, main, run_sanity_checks


def test_has_expected_callables() -> None:
    for fn in (run_sanity_checks, check_file_exists, check_csv_basic, main):
        assert callable(fn)


def test_run_sanity_checks_accepts_cfg() -> None:
    sig = inspect.signature(run_sanity_checks)
    assert "cfg" in sig.parameters


def test_check_file_exists_returns_false_for_missing(tmp_path) -> None:
    result = check_file_exists(tmp_path / "nonexistent.csv")
    assert result is False


def test_check_file_exists_returns_true_for_existing(tmp_path) -> None:
    p = tmp_path / "exists.csv"
    p.write_text("a,b\n1,2\n")
    assert check_file_exists(p) is True


def test_check_csv_basic_returns_none_for_missing(tmp_path) -> None:
    result = check_csv_basic(tmp_path / "nonexistent.csv")
    assert result is None


def test_check_csv_basic_returns_dataframe_for_valid(tmp_path) -> None:
    import pandas as pd

    p = tmp_path / "data.csv"
    p.write_text("tx_id,hh,active_power_kW\nA,2022-01-01,1.0\nB,2022-01-02,2.0\n")
    df = check_csv_basic(p, nrows=10)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["tx_id", "hh", "active_power_kW"]
