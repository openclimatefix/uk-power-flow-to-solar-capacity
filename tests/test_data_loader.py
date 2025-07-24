import os
import sys

import pandas as pd

from src.data_loader import load_csv_data, load_era5_data


def test_load_csv_data_success(setup_test_data, test_config):
    """Tests the successful loading and basic processing of CSV files."""
    paths = test_config['paths']
    power_cols = test_config['data_ingestion_params']['power_csv_cols']

    df_power, df_sites = load_csv_data(paths['power_flow_path'], paths['sites_path'], power_cols)

    assert df_power is not None
    assert not df_power.empty
    assert set(df_power.columns) == {'timestamp', 'power', 'tx_id'}
    assert pd.api.types.is_datetime64_any_dtype(df_power['timestamp'])
    assert pd.api.types.is_numeric_dtype(df_power['power'])
    assert df_power['power'].iloc[0] == 3.610

    assert df_sites is not None
    assert not df_sites.empty
    assert 'SiteName' in df_sites.columns


def test_load_csv_data_file_not_found(test_config):
    """Tests that the function handles a missing file gracefully."""
    paths = test_config['paths']
    power_cols = test_config['data_ingestion_params']['power_csv_cols']

    df_power, df_sites = load_csv_data('non_existent_file.csv', paths['sites_path'], power_cols)

    assert df_power is None
    assert df_sites is None


def test_load_era5_data_success(setup_test_data, test_config):
    """Tests the successful loading of ERA5 NetCDF data."""
    paths = test_config['paths']

    ds_era5 = load_era5_data(paths['era5_extract_dir'], paths['skt_files_path'])

    assert ds_era5 is not None
    assert 't2m' in ds_era5.data_vars
    assert 'latitude' in ds_era5.coords
    assert 'longitude' in ds_era5.coords
    assert ds_era5['time'].size > 0
