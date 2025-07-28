import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xgboost as xgb
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def test_config():
    config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def setup_test_data(tmpdir_factory, test_config):
    temp_dir = tmpdir_factory.mktemp("test_data")
    paths = test_config["paths"]

    paths["power_flow_path"] = str(temp_dir.join(os.path.basename(paths["power_flow_path"])))
    paths["sites_path"] = str(temp_dir.join(os.path.basename(paths["sites_path"])))
    paths["era5_extract_dir"] = str(temp_dir.join("era5_extracted"))
    os.makedirs(paths["era5_extract_dir"], exist_ok=True)

    power_data = {
        "timestamp": ["2012-01-01 00:00:00"],
        "tx_id": ["aldreth_primary_11kv_t1"],
        "active_power_mw": [3.610],
    }
    pd.DataFrame(power_data).to_csv(paths["power_flow_path"], index=False)

    sites_data = {"SiteName": ["ALDRETH PRIMARY 33kV"], "Easting": [544836], "Northing": [273370]}
    pd.DataFrame(sites_data).to_csv(paths["sites_path"], index=False)

    time_index = pd.to_datetime(pd.date_range("2021-01-01", periods=48, freq="h"))
    ds = xr.Dataset(
        {
            "tcc": (("time",), np.random.rand(48)),
            "ssrd": (("time",), np.random.rand(48) * 1000),
            "t2m": (("time",), np.linspace(270, 275, 48)),
            "skt": (("time",), np.linspace(268, 273, 48)),
        },
        coords={"time": time_index, "latitude": [52.34], "longitude": [0.12]},
    )
    ds.to_netcdf(os.path.join(paths["era5_extract_dir"], "test_era5.nc"))

    return test_config


@pytest.fixture
def preprocessed_df():
    dates = pd.to_datetime(pd.date_range("2022-01-01", periods=100, freq="h"))
    idx = pd.MultiIndex.from_product(
        [["site_a"], dates],
        names=["site_id", "datetime"],
    )
    df = pd.DataFrame(index=idx)
    df["power"] = np.random.rand(100) * 100
    df["tcc"] = np.random.rand(100)
    df["t2m"] = np.linspace(273, 283, 100)
    df["skt"] = np.linspace(272, 282, 100)
    df["ssrd"] = np.random.rand(100) * 1000
    return df


@pytest.fixture
def mock_feature_data(test_config):
    params = test_config["scenario_analysis"]["analogy_params"]
    idx = pd.MultiIndex.from_product(
        [["site_a"], pd.to_datetime(pd.date_range("2021-01-01", "2024-03-31", freq="h"))],
        names=["site_id", "datetime"],
    )
    num_rows = len(idx)
    X = pd.DataFrame(index=idx, data={
        "tcc_lag_1h": np.random.rand(num_rows),
        "t2m_lag_1h": np.random.uniform(270, 300, num_rows),
        "ssrd_lag_1h": np.random.rand(num_rows) * 3_000_000,
    })

    X.iloc[10, X.columns.get_loc("tcc_lag_1h")] = params["low_sun_tcc_threshold"] + 0.01
    X.iloc[10, X.columns.get_loc("ssrd_lag_1h")] = params["low_sun_ssrd_threshold"] - 1
    X.iloc[10, X.columns.get_loc("t2m_lag_1h")] = params["low_sun_t2m_k_threshold"] - 1

    X.iloc[20, X.columns.get_loc("tcc_lag_1h")] = params["high_sun_tcc_threshold"] - 0.01
    X.iloc[20, X.columns.get_loc("ssrd_lag_1h")] = params["high_sun_ssrd_threshold"] + 1
    X.iloc[20, X.columns.get_loc("t2m_lag_1h")] = params["high_sun_t2m_k_threshold"] + 1

    y = pd.Series(index=idx, data=np.random.rand(num_rows) * 100, name="power")
    return X, y

@pytest.fixture
def simple_trained_model(mock_feature_data):
    """Creates a simple, trained XGBoost model for use in tests."""
    X, y = mock_feature_data
    model = xgb.XGBRegressor(n_estimators=2, random_state=42)
    model.fit(X.iloc[:100], y.iloc[:100])
    return model
