"""Module to simulate "Low Sun" and "High Sun" scenarios using Historical Analogies.

This method avoids creating purely artificial data.
Obtains real, historical hours that match "good" or "bad" weather criteria and
transplants weather-related features into the test set to create hybrid
scenarios.

The model's predictions on this hybrid data reveal its response
to historically-grounded extreme weather, hence permitting for estimation of
embedded solar capacity.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from xgboost import XGBRegressor


def _find_weather_analogs(
    historical_features: pd.DataFrame, config: dict[str, Any], scenario_type: str,
) -> pd.DataFrame:
    """Filters the feature set to create a library of weather analogs."""
    params = config["historical_analogy_params"]
    logging.info(f"Searching for historical {scenario_type} weather analogs...")

    if scenario_type == "low_sun":
        mask = (
            (historical_features["tcc_lag_1h"] >= params["low_sun_tcc_threshold"])
            & (historical_features["ssrd_lag_1h"] <= params["low_sun_ssrd_threshold"])
            & (historical_features["t2m_lag_1h"] <= params["low_sun_t2m_k_threshold"])
        )
    elif scenario_type == "high_sun":
        mask = (
            (historical_features["tcc_lag_1h"] <= params["high_sun_tcc_threshold"])
            & (historical_features["ssrd_lag_1h"] >= params["high_sun_ssrd_threshold"])
            & (historical_features["t2m_lag_1h"] >= params["high_sun_t2m_k_threshold"])
        )
    else:
        raise ValueError("Invalid scenario_type specified.")

    analogs = historical_features[mask].copy()

    if analogs.empty:
        raise ValueError(f"No historical data found for {scenario_type} criteria.")

    analogs["hour"] = analogs.index.get_level_values("datetime").hour
    analogs["dayofweek"] = analogs.index.get_level_values("datetime").dayofweek
    logging.info(f"Found {len(analogs)} historical {scenario_type} analog hours.")
    return analogs


def _create_hybrid_row(
    target_row: pd.DataFrame,
    analogs: pd.DataFrame,
    weather_cols: list[str],
    random_state: int,
) -> pd.DataFrame:
    """Creates a hybrid feature row by transplanting weather features from an analog."""
    target_hour = target_row.index.get_level_values("datetime").hour[0]
    target_dow = target_row.index.get_level_values("datetime").dayofweek[0]

    matching_analogs = analogs[
        (analogs["hour"] == target_hour) & (analogs["dayofweek"] == target_dow)
    ]
    if matching_analogs.empty:
        matching_analogs = analogs

    chosen_analog = matching_analogs.sample(1, random_state=random_state)
    target_row[weather_cols] = chosen_analog[weather_cols].values

    for col in target_row.columns:
        if "_x_" in col and any(p in col for p in ["tcc", "ssrd", "t2m", "skt"]):
            parts = col.split("_x_")
            if parts[0] in target_row.columns and parts[1] in target_row.columns:
                target_row[col] = target_row[parts[0]] * target_row[parts[1]]

    return target_row


def _generate_scenario_predictions(
    X_test_slice: pd.DataFrame,
    model: XGBRegressor,
    analogs: pd.DataFrame,
    weather_cols: list[str],
    config: dict[str, Any],
) -> pd.Series:
    """Generates predictions for a single scenario by iterating hour-by-hour."""
    params = config["historical_analogy_params"]
    scenario_predictions = []
    for i in range(len(X_test_slice)):
        target_row = X_test_slice.iloc[[i]].copy()
        hybrid_row = _create_hybrid_row(target_row, analogs, weather_cols, i)
        prediction = model.predict(hybrid_row)[0]
        scenario_predictions.append(prediction)

    scenario_series = pd.Series(scenario_predictions, index=X_test_slice.index)
    return scenario_series.rolling(
        window=params["smoothing_window"], center=True, min_periods=1,
    ).mean()


def _calculate_and_log_capacity(
    y_low_sun: pd.Series, y_high_sun: pd.Series, config: dict[str, Any],
) -> float:
    """Calculates and logs the final embedded capacity estimate."""
    params = config["historical_analogy_params"]
    start_hour, end_hour = params["midday_start_hour"], params["midday_end_hour"]

    midday_low_sun = y_low_sun.between_time(f"{start_hour}:00", f"{end_hour}:00")
    midday_high_sun = y_high_sun.between_time(f"{start_hour}:00", f"{end_hour}:00")

    delta = midday_low_sun - midday_high_sun
    capacity_estimate = np.maximum(0, delta).mean()

    logging.info("\n\n--- Embedded Solar Capacity (Historical Analogy Method) ---")
    logging.info(f"Estimated Capacity: {capacity_estimate:.3f} MW")
    return capacity_estimate


def _plot_analogy_results(
    results: dict[str, pd.Series], site_id: str, config: dict[str, Any],
) -> None:
    """Generates and saves a plot comparing all scenario results."""
    output_dir = config["plotting"].get("plot_output_dir", "output_plots")
    site_name = site_id.replace("_primary_11kv_t1", "").replace("_", " ").title()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(
        results["actual"].index.get_level_values("datetime"),
        results["actual"].values,
        ".-", color="black", label="Actual Power",
    )
    ax.plot(
        results["baseline"].index.get_level_values("datetime"),
        results["baseline"].values,
        "x--", color="darkorange", label="Predicted (Baseline)",
    )
    ax.plot(
        results["low_sun"].index.get_level_values("datetime"),
        results["low_sun"].values,
        "-", color="mediumseagreen", label="Predicted (Low Sun Scenario)",
    )
    ax.plot(
        results["high_sun"].index.get_level_values("datetime"),
        results["high_sun"].values,
        "-", color="deepskyblue", label="Predicted (High Sun Scenario)",
    )

    ax.set_title(f"Historical Analogy Scenarios for {site_name}", fontsize=18)
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    save_path = f"{output_dir}/analogy_scenarios_plot_{site_id}.png"
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Historical analogy plot saved to {save_path}")


def run_historical_analogy_analysis(
    model: XGBRegressor,
    X_historical: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> None:
    """Runs the full Historical Analogy scenario pipeline."""
    logging.info("--- STAGE: SCENARIO ANALYSIS (HISTORICAL ANALOGY) STARTED ---")
    params = config["historical_analogy_params"]
    np.random.seed(params["random_seed"])

    low_sun_analogs = _find_weather_analogs(X_historical, config, "low_sun")
    high_sun_analogs = _find_weather_analogs(X_historical, config, "high_sun")
    weather_cols = [
        col
        for col in X_historical.columns
        if any(col.startswith(p) for p in params["weather_column_prefixes"])
    ]

    site_id = params["target_site_id"]
    y_test_site = y_test[y_test.index.get_level_values("site_id") == site_id]
    X_test_site = X_test.loc[y_test_site.index]
    duration = params["comparison_duration_hours"]

    if len(y_test_site) < duration:
        raise ValueError(f"Test set for {site_id} is not long enough.")

    X_test_slice = X_test_site.iloc[:duration]
    y_test_slice = y_test_site.iloc[:duration]

    y_pred_low_sun = _generate_scenario_predictions(
        X_test_slice, model, low_sun_analogs, weather_cols, config,
    )
    y_pred_high_sun = _generate_scenario_predictions(
        X_test_slice, model, high_sun_analogs, weather_cols, config,
    )
    y_pred_baseline = pd.Series(model.predict(X_test_slice), index=X_test_slice.index)

    _calculate_and_log_capacity(y_pred_low_sun, y_pred_high_sun, config)

    if config["plotting"].get("save_plots", False):
        results = {
            "actual": y_test_slice,
            "baseline": y_pred_baseline,
            "low_sun": y_pred_low_sun,
            "high_sun": y_pred_high_sun,
        }
        _plot_analogy_results(results, site_id, config)
