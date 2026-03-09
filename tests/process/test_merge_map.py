"""Tests for src.process.merge_map."""

from __future__ import annotations

import inspect

from process.merge_map import create_combined_power_weather_parquet, main


def test_has_expected_callables() -> None:
    for fn in (create_combined_power_weather_parquet, main):
        assert callable(fn)


def test_create_combined_signature() -> None:
    sig = inspect.signature(create_combined_power_weather_parquet)
    for param in ["power_path", "mapping_path", "weather_glob", "output_parquet", "time_coord"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_create_combined_time_coord_default() -> None:
    sig = inspect.signature(create_combined_power_weather_parquet)
    assert sig.parameters["time_coord"].default == "time"
