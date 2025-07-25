# Feature engineering process

import logging

import holidays
import numpy as np
import pandas as pd


def create_cyclical_features(df):
    """Creates cyclical (sine/cosine) features from the datetime index."""
    datetime_index = df.index.get_level_values('datetime')

    df['hour'] = datetime_index.hour
    df['dayofweek'] = datetime_index.dayofweek
    df['dayofyear'] = datetime_index.dayofyear
    df['month'] = datetime_index.month
    df['quarter'] = datetime_index.quarter
    df['weekofyear'] = datetime_index.isocalendar().week.astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)

    days_in_year = pd.Series(datetime_index.is_leap_year, index=df.index).map({True: 366, False: 365})
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / days_in_year)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / days_in_year)

    return df


def create_event_features(df):
    """Creates features based on holidays and specific days of the week."""
    datetime_index = df.index.get_level_values('datetime')
    min_year, max_year = datetime_index.min().year, datetime_index.max().year

    uk_holidays = holidays.UK(years=range(min_year, max_year + 1))

    normalized_dates = datetime_index.normalize()
    df['is_holiday'] = pd.Series(normalized_dates.isin(uk_holidays), index=df.index).astype(int)
    df['is_day_before_holiday'] = df['is_holiday'].shift(-24, fill_value=0)
    df['is_day_after_holiday'] = df['is_holiday'].shift(24, fill_value=0)

    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)

    df['hour_sin_x_is_holiday'] = df['hour_sin'] * df['is_holiday']
    df['hour_cos_x_is_holiday'] = df['hour_cos'] * df['is_holiday']
    df['hour_sin_x_is_monday'] = df['hour_sin'] * df['is_monday']
    df['hour_cos_x_is_monday'] = df['hour_cos'] * df['is_monday']

    return df


def create_lag_and_roll_features(df, weather_vars, weather_lags, weather_roll_windows):
    """Creates lagged and rolling window features for weather variables."""
    for var in weather_vars:
        if var not in df.columns:
            logging.warning("Weather variable '%s' not found. Skipping its features.", var)
            continue

        for lag in weather_lags:
            df[f'{var}_lag_{lag}h'] = df[var].shift(lag)

        for window in weather_roll_windows:
            shifted_series = df[var].shift(1)
            df[f'{var}_roll_mean_{window}h'] = shifted_series.rolling(window=window, min_periods=1).mean()
            df[f'{var}_roll_std_{window}h'] = shifted_series.rolling(window=window, min_periods=1).std()

        df[f'{var}_diff_1h'] = df[var].diff(1)

    return df


def create_power_weather_interactions(df, target_col, tcc_var_name, power_lags, power_windows, weather_lags):
    """Creates interaction features between power (target) and weather (TCC)."""
    for p_lag in power_lags:
        power_lagged = df[target_col].shift(p_lag)
        for w_lag in weather_lags:
            tcc_col = f'{tcc_var_name}_lag_{w_lag}h'
            if tcc_col in df.columns:
                df[f'{target_col}_lag_{p_lag}h_x_{tcc_col}'] = power_lagged * df[tcc_col]

    for p_win in power_windows:
        power_roll_mean = df[target_col].rolling(window=p_win, min_periods=1).mean().shift(1)
        for w_lag in weather_lags:
            tcc_col = f'{tcc_var_name}_lag_{w_lag}h'
            if tcc_col in df.columns:
                df[f'{target_col}_roll_mean_{p_win}h_x_{tcc_col}'] = power_roll_mean * df[tcc_col]

    return df


def create_weather_enhancements(df, weather_vars_map):
    """Creates enhanced weather features like squared terms and multi-hour differences."""
    for var_name in weather_vars_map.keys():
        lag1h_var = f'{var_name}_lag_1h'
        if lag1h_var not in df.columns:
            continue

        df[f'{lag1h_var}_squared'] = df[lag1h_var]**2
        df[f'{lag1h_var}_diff_3h'] = df[lag1h_var].diff(3)
        df[f'{lag1h_var}_diff_24h'] = df[lag1h_var].diff(24)

        if 'dayofyear_sin' in df.columns:
            df[f'{lag1h_var}_x_dayofyear_sin'] = df[lag1h_var] * df['dayofyear_sin']
        if 'dayofyear_cos' in df.columns:
            df[f'{lag1h_var}_x_dayofyear_cos'] = df[lag1h_var] * df['dayofyear_cos']

    if 't2m_lag_1h' in df.columns:
        df['t2m_lag_1h_celsius'] = df['t2m_lag_1h'] - 273.15

    return df


def remove_constant_features(X):
    """Removes features with zero variance."""
    logging.info("--- Removing Constant (Zero-Variance) Features ---")
    constant_columns = X.columns[X.nunique() == 1].tolist()
    if constant_columns:
        logging.info("Identified constant columns to drop: %s", constant_columns)
        X = X.drop(columns=constant_columns)
        logging.info("Dropped %d constant column(s).", len(constant_columns))
    else:
        logging.info("No constant columns found to drop.")
    return X


def create_features_for_model(df, feature_params, weather_vars, weather_vars_map):
    """
    Main function to orchestrate all feature engineering for the master DataFrame.
    """
    if df.empty:
        logging.error("Input DataFrame is empty - cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series()

    target_col = feature_params['target_column']
    tcc_var_name = feature_params['tcc_var_name']

    all_sites_processed = []
    df_sorted = df.sort_index()

    for tx_id, group in df_sorted.groupby('tx_id'):
        logging.info("Processing base features for site: %s", tx_id)

        group = create_cyclical_features(group)
        group = create_event_features(group)
        group = create_lag_and_roll_features(
            group,
            weather_vars,
            feature_params['base_weather_lags'],
            feature_params['base_weather_roll_windows']
        )
        group = create_power_weather_interactions(
            group,
            target_col,
            tcc_var_name,
            feature_params['power_interaction_lags'],
            feature_params['power_interaction_roll_windows'],
            feature_params['interaction_weather_lags']
        )
        group = create_weather_enhancements(group, weather_vars_map)

        all_sites_processed.append(group)

    df_features = pd.concat(all_sites_processed)

    cols_to_drop = weather_vars
    df_features = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])
    df_features.dropna(inplace=True)

    if df_features.empty:
        logging.error("No data remains after all feature engineering and NaN removal.")
        return pd.DataFrame(), pd.Series()

    feature_cols = [col for col in df_features.columns if col != target_col and col != 'tx_id']
    X = df_features[feature_cols]
    y = df_features[target_col]

    # Final refinement stage
    X = remove_constant_features(X)

    logging.info("Feature engineering complete. Final shape of X: %s, y: %s", X.shape, y.shape)
    logging.info("Total number of features: %d", len(X.columns))

    return X, y
