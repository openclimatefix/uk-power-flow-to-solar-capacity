
import pandas as pd

from src.data_loader import load_csv_data
from src.preprocessing import (
    get_site_coordinates,
    handle_missing_values,
    process_single_site_power,
)


def test_get_site_coordinates(setup_test_data, test_config):
    """Tests the coordinate lookup and conversion logic."""
    paths = test_config['paths']
    power_cols = test_config['data_ingestion_params']['power_csv_cols']
    _, df_sites = load_csv_data(paths['power_flow_path'], paths['sites_path'], power_cols)

    coords = get_site_coordinates('aldreth_primary_11kv_t1', df_sites)

    assert coords is not None
    assert 'latitude' in coords
    assert 'longitude' in coords
    assert abs(coords['latitude'] - 52.34) < 0.1


def test_process_single_site_power(setup_test_data, test_config):
    """Tests power data cleaning, resampling, and filtering."""
    paths = test_config['paths']
    data_cfg = test_config['data_ingestion_params']
    df_power, _ = load_csv_data(paths['power_flow_path'], paths['sites_path'], data_cfg['power_csv_cols'])

    start_dt = pd.to_datetime(data_cfg['analysis_start_date'], utc=True)
    end_dt = pd.to_datetime(data_cfg['analysis_end_date'], utc=True)

    power_series = process_single_site_power('aldreth_primary_11kv_t1', df_power, start_dt, end_dt)

    assert power_series is not None
    assert not power_series.empty
    assert (power_series >= 0).all()
    assert isinstance(power_series.index, pd.DatetimeIndex)


def test_handle_missing_values():
    """Tests the NaN interpolation logic."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:00', '2023-01-01 13:00', '2023-01-01 14:00']),
        'power': [100.0, None, 120.0],
        't2m': [275.0, 276.0, 277.0],
        'skt': [None, None, None],
        'tx_id': ['site_a', 'site_a', 'site_a']
    }
    df_with_nans = pd.DataFrame(data).set_index('timestamp')

    df_clean = handle_missing_values(df_with_nans, era5_vars=['t2m', 'skt'])

    assert not df_clean.isnull().values.any()
    assert df_clean['power'].iloc[1] == 110.0
    assert (df_clean['skt'] == df_clean['t2m']).all()
