import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_config():
    """
    Loads the test configuration from the dedicated test_config.yaml file.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def setup_test_data(tmpdir_factory, test_config):
    """
    Creates realistic test data files in a temporary directory based on the test config.
    This fixture runs only once per test session.
    """
    temp_dir = tmpdir_factory.mktemp("test_data")
    paths = test_config['paths']

    paths['power_flow_path'] = os.path.join(temp_dir, os.path.basename(paths['power_flow_path']))
    paths['sites_path'] = os.path.join(temp_dir, os.path.basename(paths['sites_path']))
    paths['era5_extract_dir'] = os.path.join(temp_dir, 'era5_extracted')
    os.makedirs(paths['era5_extract_dir'], exist_ok=True)

    # Create Power Data CSV
    power_data = {
        'timestamp': ['2021-01-01 00:00:00', '2021-01-01 00:30:00'],
        'tx_id': ['aldreth_primary_11kv_t1', 'aldreth_primary_11kv_t1'],
        'active_power_mw': [3.610, 3.825]
    }
    pd.DataFrame(power_data).to_csv(paths['power_flow_path'], index=False)

    # Create Sites Data CSV
    sites_data = {'SiteName': ['ALDRETH PRIMARY 33kV'], 'Easting': [544836], 'Northing': [273370]}
    pd.DataFrame(sites_data).to_csv(paths['sites_path'], index=False)

    # Create ERA5 Data NetCDF
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
    ds.to_netcdf(os.path.join(paths['era5_extract_dir'], "test_era5.nc"))

    return test_config


@pytest.fixture
def mock_feature_data():
    """
    Creates a larger, synthetic DataFrame for testing modeling functions
    that require a longer time series.
    """
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
