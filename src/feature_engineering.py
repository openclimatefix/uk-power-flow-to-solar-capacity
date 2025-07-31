"""Functions for creating features for the power demand model."""

import logging
from typing import Any

import holidays
import numpy as np
import pandas as pd


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates cyclical (sine/cosine) features from the datetime index."""
    datetime_index = df.index.get_level_values("datetime")

    df["hour"] = datetime_index.hour
    df["dayofweek"] = datetime_index.dayofweek
    df["dayofyear"] = datetime_index.dayofyear
    df["month"] = datetime_index.month
    df["quarter"] = datetime_index.quarter
    df["weekofyear"] = datetime_index.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    days_in_year = pd.Series(datetime_index.is_leap_year, index=df.index).map(
        {True: 366, False: 365},
    )
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / days_in_year)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / days_in_year)

    # Scientifically backed solar position features based on solar geometry
    # Solar declination angle - Earth's tilt effect (fundamental for PV calculations)
    day_of_year = datetime_index.dayofyear
    solar_declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    df['solar_declination'] = solar_declination
    
    # Hour angle from solar noon - Earth's rotation effect
    hour_angle = 15 * (datetime_index.hour - 12)  # 15 degrees per hour
    df['hour_angle'] = hour_angle
    
    # Solar elevation angle for UK (latitude ~52°N) - determines solar irradiance potential
    uk_latitude = 52.0
    lat_rad = np.radians(uk_latitude)
    dec_rad = np.radians(solar_declination)
    hour_rad = np.radians(hour_angle)
    
    # Solar elevation calculation from celestial mechanics
    sin_elevation = (np.sin(lat_rad) * np.sin(dec_rad) + 
                    np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad))
    
    # Clip to valid range [-1, 1] to avoid numerical errors
    sin_elevation = np.clip(sin_elevation, -1, 1)
    solar_elevation = np.degrees(np.arcsin(sin_elevation))
    
    df['solar_elevation'] = solar_elevation
    df['is_sun_above_horizon'] = (solar_elevation > 0).astype(int)
    df['is_meaningful_solar'] = (solar_elevation > 10).astype(int)  # >10° for significant PV output
    
    # Air mass coefficient - atmospheric effect on solar irradiance
    zenith_angle = 90 - solar_elevation
    air_mass = np.where(solar_elevation > 0, 
                       1 / np.cos(np.radians(zenith_angle)), 
                       10)  # Default high value when sun is down
    df['air_mass'] = np.clip(air_mass, 1, 10)  # Cap at reasonable range

    return df


def create_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on holidays and specific days of the week."""
    datetime_index = df.index.get_level_values("datetime")
    min_year, max_year = datetime_index.min().year, datetime_index.max().year

    uk_holidays = holidays.UK(years=range(min_year, max_year + 1))

    normalized_dates = datetime_index.normalize()
    df["is_holiday"] = pd.Series(
        normalized_dates.isin(uk_holidays), index=df.index,
    ).astype(int)
    df["is_day_before_holiday"] = df["is_holiday"].shift(-24, fill_value=0)
    df["is_day_after_holiday"] = df["is_holiday"].shift(24, fill_value=0)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_monday"] = (df["dayofweek"] == 0).astype(int)

    df["hour_sin_x_is_holiday"] = df["hour_sin"] * df["is_holiday"]
    df["hour_cos_x_is_holiday"] = df["hour_cos"] * df["is_holiday"]
    df["hour_sin_x_is_monday"] = df["hour_sin"] * df["is_monday"]
    df["hour_cos_x_is_monday"] = df["hour_cos"] * df["is_monday"]

    return df


def create_lag_and_roll_features(
    df: pd.DataFrame,
    weather_vars: list[str],
    weather_lags: list[int],
    weather_roll_windows: list[int],
) -> pd.DataFrame:
    """Creates lagged and rolling window features for weather variables."""
    for var in weather_vars:
        if var not in df.columns:
            logging.warning("Weather variable '%s' not found. Skipping features.", var)
            continue

        for lag in weather_lags:
            df[f"{var}_lag_{lag}h"] = df[var].shift(lag)

        for window in weather_roll_windows:
            shifted_series = df[var].shift(1)
            rolling_op = shifted_series.rolling(window=window, min_periods=1)
            df[f"{var}_roll_mean_{window}h"] = rolling_op.mean()
            df[f"{var}_roll_std_{window}h"] = rolling_op.std()

        df[f"{var}_diff_1h"] = df[var].diff(1)

    return df


def create_power_features(
    df: pd.DataFrame, target_col: str, power_lags: list[int], power_windows: list[int],
) -> pd.DataFrame:
    """Creates lagged and rolling window features for the power target column."""
    if target_col not in df.columns:
        logging.warning("Target column '%s' not found for power features.", target_col)
        return df

    for lag in power_lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    for window in power_windows:
        shifted_series = df[target_col].shift(1)
        rolling_op = shifted_series.rolling(window=window, min_periods=1)
        df[f"{target_col}_roll_mean_{window}h"] = rolling_op.mean()
        df[f"{target_col}_roll_std_{window}h"] = rolling_op.std()

    df[f"{target_col}_diff_1h"] = df[target_col].diff(1)

    return df


def create_power_weather_interactions(
    df: pd.DataFrame,
    target_col: str,
    tcc_var_name: str,
    power_lags: list[int],
    power_windows: list[int],
    weather_lags: list[int],
) -> pd.DataFrame:
    """Creates interaction features between power (target) and weather (TCC)."""
    for p_lag in power_lags:
        power_lagged = df[target_col].shift(p_lag)
        for w_lag in weather_lags:
            tcc_col = f"{tcc_var_name}_lag_{w_lag}h"
            if tcc_col in df.columns:
                df[f"{target_col}_lag_{p_lag}h_x_{tcc_col}"] = power_lagged * df[tcc_col]

    for p_win in power_windows:
        power_roll_mean = df[target_col].rolling(window=p_win, min_periods=1).mean().shift(1)
        for w_lag in weather_lags:
            tcc_col = f"{tcc_var_name}_lag_{w_lag}h"
            if tcc_col in df.columns:
                df[f"{target_col}_roll_mean_{p_win}h_x_{tcc_col}"] = (
                    power_roll_mean * df[tcc_col]
                )

    return df


def create_weather_enhancements(
    df: pd.DataFrame, weather_vars_map: dict[str, Any],
) -> pd.DataFrame:
    """Creates enhanced weather features like squared terms and multi-hour differences."""
    for var_name in weather_vars_map:
        lag1h_var = f"{var_name}_lag_1h"
        if lag1h_var not in df.columns:
            continue

        df[f"{lag1h_var}_squared"] = df[lag1h_var] ** 2
        df[f"{lag1h_var}_diff_3h"] = df[lag1h_var].diff(3)
        df[f"{lag1h_var}_diff_24h"] = df[lag1h_var].diff(24)

        if "dayofyear_sin" in df.columns:
            df[f"{lag1h_var}_x_dayofyear_sin"] = df[lag1h_var] * df["dayofyear_sin"]
        if "dayofyear_cos" in df.columns:
            df[f"{lag1h_var}_x_dayofyear_cos"] = df[lag1h_var] * df["dayofyear_cos"]

    if "t2m_lag_1h" in df.columns:
        df["t2m_lag_1h_celsius"] = df["t2m_lag_1h"] - 273.15

    return df


def remove_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Removes features with zero variance."""
    constant_columns = X.columns[X.nunique() == 1].tolist()
    if constant_columns:
        logging.info("Identified constant columns to drop: %s", constant_columns)
        X = X.drop(columns=constant_columns)
        logging.info("Dropped %d constant column(s).", len(constant_columns))
    else:
        logging.info("No constant columns found to drop.")
    return X


def create_features_for_model(
    df: pd.DataFrame,
    feature_params: dict[str, Any],
    weather_vars: list[str],
    weather_vars_map: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Orchestrates all feature engineering for the master DataFrame."""
    if df.empty:
        logging.error("Input DataFrame is empty - cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series()

    target_col = feature_params["target_column"]
    tcc_var_name = feature_params["tcc_var_name"]

    all_sites_processed = []
    df_sorted = df.sort_index()

    for tx_id, group in df_sorted.groupby("tx_id"):
        logging.info("Processing base features for site: %s", tx_id)

        group = create_cyclical_features(group)
        group = create_event_features(group)
        group = create_lag_and_roll_features(
            group,
            weather_vars,
            feature_params["base_weather_lags"],
            feature_params["base_weather_roll_windows"],
        )
        group = create_power_features(
            group,
            target_col,
            feature_params["power_interaction_lags"],
            feature_params["power_interaction_roll_windows"],
        )
        group = create_power_weather_interactions(
            group,
            target_col,
            tcc_var_name,
            feature_params["power_interaction_lags"],
            feature_params["power_interaction_roll_windows"],
            feature_params["interaction_weather_lags"],
        )
        group = create_weather_enhancements(group, weather_vars_map)

        all_sites_processed.append(group)

    df_features = pd.concat(all_sites_processed)

    cols_to_drop_existing = [
        col for col in weather_vars if col in df_features.columns
    ]
    df_features = df_features.drop(columns=cols_to_drop_existing)
    df_features.dropna(inplace=True)

    if df_features.empty:
        logging.error("No data remains after all feature engineering and NaN removal.")
        return pd.DataFrame(), pd.Series()

    feature_cols = [
        col for col in df_features.columns if col not in [target_col, "tx_id"]
    ]
    X = df_features[feature_cols]
    y = df_features[target_col]

    # Final refinement stage
    X = remove_constant_features(X)

    logging.info(
        "Feature engineering complete. Final shape of X: %s, y: %s", X.shape, y.shape,
    )
    logging.info("Total number of features: %d", len(X.columns))

    return X, y
