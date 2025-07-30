"""Model training, evaluation, and management functionality."""

import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def _parse_param_distributions(params_dict: dict[str, str]) -> dict[str, Any]:
    """Parses string representations of distributions into scipy.stats objects."""
    distributions = {"uniform": uniform, "randint": randint, "loguniform": loguniform}
    parsed_dict = {}
    for param, value in params_dict.items():
        for dist_name, dist_func in distributions.items():
            if value.startswith(dist_name):
                try:
                    args_str = value[len(dist_name) + 1 : -1]
                    args = [float(arg.strip()) for arg in args_str.split(",")]
                    parsed_dict[param] = dist_func(*args)
                    break
                except (ValueError, IndexError):
                    logging.error("Could not parse distribution string: %s", value)
    return parsed_dict


def _log_feature_importances(model: xgb.XGBRegressor, feature_names: list[str]) -> None:
    """Logs the top 20 feature importances from a trained model."""
    if hasattr(model, "feature_importances_"):
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_},
        ).sort_values("importance", ascending=False)
        logging.info(
            "Top 20 features from the tuned model:\n%s", feature_importance_df.head(20),
        )


def split_data(
    X: pd.DataFrame, y: pd.Series, split_dates: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Splits data into training, validation, and test sets based on dates."""
    train_end = pd.to_datetime(split_dates["train_end"], utc=True)
    val_start = pd.to_datetime(split_dates["val_start"], utc=True)
    val_end = pd.to_datetime(split_dates["val_end"], utc=True)
    test_start = pd.to_datetime(split_dates["test_start"], utc=True)

    datetime_idx = X.index.get_level_values("datetime")

    X_train = X[datetime_idx <= train_end]
    y_train = y[datetime_idx <= train_end]

    X_val = X[(datetime_idx >= val_start) & (datetime_idx <= val_end)]
    y_val = y[(datetime_idx >= val_start) & (datetime_idx <= val_end)]

    X_test = X[datetime_idx >= test_start]
    y_test = y[datetime_idx >= test_start]

    # Ensure consistent sorting
    X_train = X_train.sort_index(level=["site_id", "datetime"])
    y_train = y_train.sort_index(level=["site_id", "datetime"])
    if not X_val.empty:
        X_val = X_val.sort_index(level=["site_id", "datetime"])
        y_val = y_val.sort_index(level=["site_id", "datetime"])
    if not X_test.empty:
        X_test = X_test.sort_index(level=["site_id", "datetime"])
        y_test = y_test.sort_index(level=["site_id", "datetime"])

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator_params: dict[str, Any],
    search_params: dict[str, Any],
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    """Performs hyperparameter tuning using RandomizedSearchCV."""
    base_model = xgb.XGBRegressor(**estimator_params)

    tscv = TimeSeriesSplit(n_splits=search_params["n_cv_splits"])
    logging.info(
        "Using TimeSeriesSplit with %d splits for cross-validation.",
        search_params["n_cv_splits"],
    )

    param_dist = _parse_param_distributions(search_params["param_distributions"])

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=search_params["n_iterations"],
        scoring=search_params["scoring"],
        cv=tscv,
        random_state=search_params["random_state"],
        n_jobs=estimator_params.get("n_jobs", -1),
        verbose=search_params["verbose"],
    )

    logging.info(
        "Starting RandomizedSearchCV with %d iterations...",
        search_params["n_iterations"],
    )
    random_search.fit(X_train, y_train)

    logging.info("Best hyperparameters found: %s", random_search.best_params_)
    logging.info("Best cross-validation RMSE: %.4f", -random_search.best_score_)

    best_model = random_search.best_estimator_
    _log_feature_importances(best_model, X_train.columns.tolist())

    return best_model, random_search.best_params_


def retrain_with_constraints(
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    constraint_map: dict[str, int],
    base_params: dict[str, Any],
    interaction_constraint_features: list[str],
) -> xgb.XGBRegressor:
    """Retrains a final model with best hyperparameters and constraints."""
    logging.info("Defining final model with optimal hyperparameters and constraints.")

    feature_names = X_train.columns.tolist()

    # Create monotonic constraint tuple based on feature order
    monotone_constraints = [0] * len(feature_names)
    for feature, constraint in constraint_map.items():
        if feature in feature_names:
            feature_index = feature_names.index(feature)
            monotone_constraints[feature_index] = constraint
            logging.info("Applying monotonic constraint for '%s': %d", feature, constraint)

    final_params = base_params.copy()
    final_params.update(best_params)
    final_params["monotone_constraints"] = tuple(monotone_constraints)

    # Define interaction constraints
    if interaction_constraint_features:
        interaction_constraints = [interaction_constraint_features]
        final_params["interaction_constraints"] = interaction_constraints
        logging.info(
            "Applying interaction constraint for features: %s",
            interaction_constraint_features,
        )

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_train, y_train)

    return final_model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_set: pd.DataFrame,
    y_set: pd.Series,
    set_name: str,
    target_ids: list[str],
) -> pd.DataFrame | None:
    """Evaluates the model on a given dataset and logs metrics per site."""
    if X_set.empty or y_set.empty:
        logging.warning("%s set is empty. Skipping evaluation.", set_name.capitalize())
        return None

    y_pred = model.predict(X_set)
    results_df = pd.DataFrame({"Actual": y_set, "Predicted": y_pred}, index=y_set.index)

    for site_id in target_ids:
        if site_id not in results_df.index.get_level_values("site_id"):
            continue

        site_df = results_df.loc[site_id]
        mae = mean_absolute_error(site_df["Actual"], site_df["Predicted"])
        rmse = np.sqrt(mean_squared_error(site_df["Actual"], site_df["Predicted"]))
        r2 = r2_score(site_df["Actual"], site_df["Predicted"])

        logging.info("Metrics for %s SET - %s:", set_name.upper(), site_id)
        logging.info("  MAE: %.3f MW", mae)
        logging.info("  RMSE: %.3f MW", rmse)
        logging.info("  R-squared: %.3f", r2)

    return results_df


def save_model(model: xgb.XGBRegressor, path: str, filename: str) -> None:
    """Saves a trained model using pickle."""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved successfully to: %s", save_path)


def load_model(path: str, filename: str) -> xgb.XGBRegressor | None:
    """Loads a trained model from a pickle file."""
    load_path = os.path.join(path, filename)
    if not os.path.exists(load_path):
        logging.error("Model file not found at: %s", load_path)
        return None
    with open(load_path, "rb") as f:
        model = pickle.load(f)  # noqa: S301
    logging.info("Model loaded successfully from: %s", load_path)
    return model
