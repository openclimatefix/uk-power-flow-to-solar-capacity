from src.feature_engineering import (
    create_cyclical_features,
    create_lag_and_roll_features,
)


def test_create_cyclical_features(preprocessed_df):
    """
    Tests the creation of cyclical (sine/cosine) time-based features.
    """
    df_featured = create_cyclical_features(preprocessed_df.copy())

    assert 'hour_sin' in df_featured.columns
    assert 'dayofyear_cos' in df_featured.columns
    assert (df_featured['hour_sin'] >= -1).all() and (df_featured['hour_sin'] <= 1).all()


def test_create_lag_and_roll_features(preprocessed_df, test_config):
    """
    Tests the creation of lagged and rolling window features.
    """
    feature_params = test_config['feature_params']
    weather_vars = test_config['data_ingestion_params']['era5_vars']

    df_featured = create_lag_and_roll_features(
        preprocessed_df.copy(),
        weather_vars,
        feature_params['base_weather_lags'],
        feature_params['base_weather_roll_windows']
    )

    assert 't2m_lag_1h' in df_featured.columns
    assert 'tcc_roll_mean_3h' in df_featured.columns
