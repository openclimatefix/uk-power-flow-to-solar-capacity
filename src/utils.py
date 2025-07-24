# REFERENCE - Functionality not currently utilised

import logging
import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import plot as plt


def inspect_data_summary(all_sites_data, time_coord_name):
    if not all_sites_data:
        logging.warning("The 'all_sites_data' dictionary is empty. No data to summarize.")
        return

    for site_id, data_dict in all_sites_data.items():
        print(f"\nSite: {site_id}")

        power_series = data_dict.get('power')
        if power_series is not None:
            print(f"  Power Data: {len(power_series)} records")
            if power_series.isnull().any():
                logging.error("NaNs found in Power Data for site %s!", site_id)
        else:
            print("  Power Data: Not available")

        era5_ds = data_dict.get('era5')
        if era5_ds is not None:
            print("  ERA5 Weather Data:")
            for var_name in era5_ds.data_vars:
                print(f"    - '{var_name}': {era5_ds[var_name].size} records")
                if era5_ds[var_name].isnull().any():
                    logging.error("NaNs found in ERA5 var '%s' for site %s!", var_name, site_id)
        else:
            print("  ERA5 Weather Data: Not available")

        coords = data_dict.get('coords')
        if coords:
            print(f"  Coordinates: Lat {coords['latitude']:.4f}, Lon {coords['longitude']:.4f}")
        else:
            print("  Coordinates: Not available")


def _check_nans(X, y):
    """Validates the presence of NaN values in features and target."""
    nan_cols = X.columns[X.isnull().any()].tolist()
    if not nan_cols:
        logging.info("Validation successful: No NaN values found in feature set X.")
    else:
        logging.error("Validation failed: NaN values found in feature columns: %s", nan_cols)

    if not y.isnull().any():
        logging.info("Validation successful: No NaN values found in target variable y ('%s').", y.name)
    else:
        logging.error("Validation failed: %d NaN values found in target variable y ('%s').", y.isnull().sum(), y.name)


def _check_consistency(X, y):
    """Validates the shape and index consistency between features and target."""
    logging.info("Data shapes: X=%s, y=%s", X.shape, y.shape)
    if X.shape[0] != y.shape[0]:
        logging.error("Validation failed: Mismatch in number of samples between X and y.")
    else:
        logging.info("Validation successful: Number of samples in X and y are consistent.")

    if not X.index.equals(y.index):
        logging.error("Validation failed: Indices of X and y are not identical.")
    else:
        logging.info("Validation successful: Indices of X and y are identical.")


def _check_data_types(X):
    """Validates data types and checks for infinite values."""
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if not object_cols:
        logging.info("Validation successful: No 'object' dtype columns found in X.")
    else:
        logging.error("Validation failed: Found 'object' dtype columns: %s", object_cols)

    inf_cols = X.columns[np.isinf(X).any()].tolist()
    if not inf_cols:
        logging.info("Validation successful: No infinite values found in X.")
    else:
        logging.error("Validation failed: Found columns with infinite values: %s", inf_cols)


def _check_constant_features(X):
    """Validates for the presence of constant (zero-variance) features."""
    constant_features = X.columns[X.nunique() == 1].tolist()
    if not constant_features:
        logging.info("Validation successful: No constant features found in X.")
    else:
        logging.warning("Validation warning: Found constant features: %s", constant_features)


def _display_statistics(X, y):
    """Displays descriptive statistics for features and target."""
    print("\nDescriptive Statistics (X):")
    with pd.option_context('display.max_rows', 200, 'display.max_columns', 10, 'display.width', 120):
        print(X.describe().T[['count', 'mean', 'std', 'min', 'max']])

    print(f"\nDescriptive Statistics (y: {y.name}):")
    print(y.describe())


def run_quality_assessment(X, y):
    """
    Orchestrates a comprehensive quality check on the final feature (X)
    and target (y) sets before model training.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        logging.error("Quality assessment failed: X is not a DataFrame or y is not a Series.")
        return

    _check_nans(X, y)
    _check_consistency(X, y)
    _check_data_types(X)
    _check_constant_features(X)
    _display_statistics(X, y)


def sanity_check_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Performs a sanity check on the data splits, verifying shapes and chronological order.
    """

    def check_set(name, X, y):
        if not X.empty:
            start = X.index.get_level_values('datetime').min()
            end = X.index.get_level_values('datetime').max()
            logging.info("%s Set: %d samples from %s to %s", name, len(X), start, end)
            if len(X) != len(y):
                logging.error("%s set sample count mismatch: X=%d, y=%d", name, len(X), len(y))
        else:
            logging.warning("%s set is empty.", name)

    check_set("Training", X_train, y_train)
    check_set("Validation", X_val, y_val)
    check_set("Test", X_test, y_test)

    if not X_train.empty and not X_val.empty:
        train_end = X_train.index.get_level_values('datetime').max()
        val_start = X_val.index.get_level_values('datetime').min()
        if train_end >= val_start:
            logging.error("Chronological error: Training set ends at or after validation set begins.")
        else:
            logging.info("Chronological check OK: Training set ends before validation set begins.")

    if not X_val.empty and not X_test.empty:
        val_end = X_val.index.get_level_values('datetime').max()
        test_start = X_test.index.get_level_values('datetime').min()
        if val_end >= test_start:
            logging.error("Chronological error: Validation set ends at or after test set begins.")
        else:
            logging.info("Chronological check OK: Validation set ends before test set begins.")


def log_and_plot_feature_importance(model, feature_names, plot_config):
    """
    Logs and plots the top feature importances from a trained model.
    """
    if not hasattr(model, 'feature_importances_'):
        logging.error("Model does not have 'feature_importances_' attribute.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    sorted_importances = importances.sort_values(ascending=False)

    logging.info("Top 20 feature importances:\n%s", sorted_importances.head(20))

    if plot_config.get('save_plots', False):
        top_n = 20
        plot_importances = sorted_importances.head(top_n)

        plt.figure(figsize=(12, 8))
        plot_importances.sort_values(ascending=True).plot(kind='barh', color='skyblue')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        output_dir = plot_config.get('plot_output_dir', 'output_plots')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'feature_importance.png')

        plt.savefig(save_path)
        plt.close()
        logging.info("Feature importance plot saved to: %s", save_path)


def plot_predictions(results_df, set_name, target_ids, plot_config):
    """
    Plots actual vs. predicted values for a specific date range for each site.
    """
    if not plot_config.get('save_plots', False) or results_df is None:
        return

    logging.info("--- Generating Prediction Plots for %s SET ---", set_name.upper())

    plot_start = plot_config.get('plot_start_date', '2024-01-02')
    plot_end = plot_config.get('plot_end_date', '2024-01-08')
    output_dir = plot_config.get('plot_output_dir', 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    for site_id in target_ids:
        if site_id not in results_df.index.get_level_values('site_id'):
            continue

        site_df = results_df.loc[site_id].loc[plot_start:plot_end]
        if site_df.empty:
            logging.warning("No data found for site %s in plot range %s to %s. Skipping plot.", site_id, plot_start, plot_end)
            continue

        site_name_display = site_id.replace("_primary_11kv_t1", "").replace("_", " ").title()

        plt.figure(figsize=(15, 7))
        plt.plot(site_df.index, site_df['Actual'], label='Actual Power', marker='.', linestyle='-')
        plt.plot(site_df.index, site_df['Predicted'], label=f'Predicted Power ({set_name})', marker='x', linestyle='--')

        plt.title(f'{set_name.upper()}: Actual vs. Predicted - {site_name_display}\n({plot_start} to {plot_end})')
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %b %d'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{set_name.lower()}_prediction_{site_id}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info("Prediction plot for site %s saved to: %s", site_id, save_path)
