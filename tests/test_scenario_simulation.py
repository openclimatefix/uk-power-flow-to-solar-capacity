import pandas as pd
import pytest

from src.scenario_generation_a import (
    _define_extreme_scenarios,
)
from src.scenario_generation_a import (
    run_scenario_analysis as run_overwrite_analysis,
)
from src.scenario_generation_b import (
    _find_weather_analogs,
    run_historical_analogy_analysis,
)


def test_overwrite_define_scenarios(preprocessed_df, test_config):
    """
    Tests that the weather overwrite method correctly defines scenarios.
    """
    cfg = test_config.copy()
    cfg["data_ingestion_params"]["weather_vars_map"] = {"ssrd": "ssrd", "tcc": "tcc"}
    cfg["scenario_analysis_params"] = cfg["scenario_analysis"]["overwrite_params"]

    site_df = preprocessed_df.loc["site_a"]
    ds = site_df.to_xarray().rename({"datetime": "time"})
    scenarios = _define_extreme_scenarios(ds, cfg)

    assert "HighSun" in scenarios
    assert "LowSun" in scenarios
    assert isinstance(scenarios["HighSun"]["ssrd"], float)
    assert scenarios["LowSun"]["tcc"] == 1.0


def test_run_weather_overwrite_analysis(
    simple_trained_model, preprocessed_df, test_config,
):
    """
    Integration test for the weather overwrite analysis.
    """
    cfg = test_config.copy()
    cfg["scenario_analysis_params"] = cfg["scenario_analysis"]["overwrite_params"]

    master_df = preprocessed_df.reset_index().rename(columns={"site_id": "tx_id"})
    master_df["datetime"] = pd.to_datetime(master_df["datetime"]).dt.tz_localize("UTC")
    master_df = master_df.set_index("datetime")

    try:
        run_overwrite_analysis(simple_trained_model, master_df, cfg)
    except Exception as e:
        pytest.fail(f"Weather overwrite analysis failed with an exception: {e}")


def test_analogy_find_analogs(mock_feature_data, test_config):
    """
    Tests that the historical analogy method correctly filters for analogs.
    """
    X, _ = mock_feature_data
    cfg = test_config.copy()
    cfg["historical_analogy_params"] = cfg["scenario_analysis"]["analogy_params"]
    params = cfg["historical_analogy_params"]

    low_sun_analogs = _find_weather_analogs(X, cfg, "low_sun")
    assert not low_sun_analogs.empty
    assert (low_sun_analogs["tcc_lag_1h"] >= params["low_sun_tcc_threshold"]).all()

    high_sun_analogs = _find_weather_analogs(X, cfg, "high_sun")
    assert not high_sun_analogs.empty
    assert (high_sun_analogs["tcc_lag_1h"] <= params["high_sun_tcc_threshold"]).all()


def test_run_historical_analogy_analysis(
    simple_trained_model, mock_feature_data, test_config,
):
    """
    Integration test for the historical analogy analysis.
    """
    X, y = mock_feature_data
    X_hist_small = X[X.index.get_level_values("datetime").year == 2021]
    X_test_small = X[X.index.get_level_values("datetime").year == 2022]
    y_test_small = y[y.index.get_level_values("datetime").year == 2022]

    cfg = test_config.copy()
    cfg["historical_analogy_params"] = cfg["scenario_analysis"]["analogy_params"]
    try:
        run_historical_analogy_analysis(
            simple_trained_model, X_hist_small, X_test_small, y_test_small, cfg,
        )
    except Exception as e:
        pytest.fail(f"Historical analogy analysis failed with an exception: {e}")
