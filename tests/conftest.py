
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_data_dir(tmpdir_factory):
    """Create a temporary directory for test data."""
    return tmpdir_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_config(test_data_dir):
    """Create and load the test configuration YAML file."""
    config_content = f"""
project_name: 'power-forecasting-test'
paths:
  power_flow_path: '{test_data_dir}/power.csv'
  sites_path: '{test_data_dir}/sites.csv'
  era5_extract_dir: '{test_data_dir}/era5_extracted/'
  skt_files_path: '{test_data_dir}/skt_*.nc'
data_ingestion_params:
  target_transformer_ids: ['aldreth_primary_11kv_t1']
  analysis_start_date: '2021-01-01'
  analysis_end_date: '2021-01-02'
  power_csv_cols:
    timestamp: 'timestamp'
    active_power_mw: 'power'
    tx_id: 'tx_id'
  era5_vars: ['tcc', 'ssrd', 't2m', 'skt']
feature_params:
  target_column: 'power'
    """
    config_path = test_data_dir.join("test_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def setup_test_data(test_data_dir):
    """Create realistic test CSV and NetCDF files based on notebook output."""
    # Power Data
    power_data = {
        'timestamp': ['2021-01-01 00:00:00', '2021-01-01 00:30:00'],
        'tx_id': ['aldreth_primary_11kv_t1', 'aldreth_primary_11kv_t1'],
        'active_power_mw': [3.610, 3.825]
    }
    pd.DataFrame(power_data).to_csv(f"{test_data_dir}/power.csv", index=False)

    # Sites Data
    sites_data = {'SiteName': ['ALDRETH PRIMARY 33kV'], 'Easting': [544836], 'Northing': [273370]}
    pd.DataFrame(sites_data).to_csv(f"{test_data_dir}/sites.csv", index=False)

    # ERA5 Data (create a dummy .nc file)
    extract_dir = test_data_dir.mkdir("era5_extracted")

    time_index = pd.to_datetime(pd.date_range('2021-01-01', periods=48, freq='h'))

    ds = xr.Dataset(
        {
            'tcc': (('time',), np.random.rand(48)),
            'ssrd': (('time',), np.random.rand(48) * 1000),
            't2m': (('time',), np.linspace(270, 275, 48)),
            'skt': (('time',), np.linspace(268, 273, 48)),
        },
        coords={'time': time_index, 'latitude': [52.34], 'longitude': [0.12]}
    )
    ds.to_netcdf(f"{extract_dir}/test_era5.nc")

    return test_data_dir


@pytest.fixture
def mock_feature_data():
    """Creates a realistic, multi-indexed DataFrame for testing model functions."""
    idx = pd.MultiIndex.from_product(
        [['site_a'], pd.to_datetime(pd.date_range('2021-01-01', '2024-03-31', freq='h', tz='UTC'))],
        names=['site_id', 'datetime']
    )
    X = pd.DataFrame(index=idx, data={
        'feature1': range(len(idx)), 
        'tcc_lag_1h': range(len(idx)),
        't2m_lag_1h': range(len(idx)),
        'ssrd_lag_1h': range(len(idx))
    })
    y = pd.Series(index=idx, data=range(len(idx)), name='power')
    return X, y
