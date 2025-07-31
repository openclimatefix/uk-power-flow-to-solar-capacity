# XGBoost only - including intermediate stages

import logging
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def _parse_param_distributions(params_dict):
    """Parses string representations of distributions into scipy.stats objects."""
    parsed_dict = {}
    for param, value in params_dict.items():
        parsed_dict[param] = eval(value, {"uniform": uniform, "randint": randint, "loguniform": loguniform})
    return parsed_dict


def _log_feature_importances(model, feature_names):
    """Logs the top 20 feature importances from a trained model."""
    if hasattr(model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logging.info("Top 20 features from the tuned model:\n%s", feature_importance_df.head(20))


def split_data(X, y, split_dates):
    """
    Splits data into training, validation, and test sets based on dates.
    """
    train_end = pd.to_datetime(split_dates['train_end'], utc=True)
    val_start = pd.to_datetime(split_dates['val_start'], utc=True)
    val_end = pd.to_datetime(split_dates['val_end'], utc=True)
    test_start = pd.to_datetime(split_dates['test_start'], utc=True)

    datetime_idx = X.index.get_level_values('datetime')

    X_train = X[datetime_idx <= train_end].sort_index(level=['site_id', 'datetime'])
    y_train = y[datetime_idx <= train_end].sort_index(level=['site_id', 'datetime'])

    X_val = X[(datetime_idx >= val_start) & (datetime_idx <= val_end)]
    y_val = y[(datetime_idx >= val_start) & (datetime_idx <= val_end)]
    if not X_val.empty:
        X_val = X_val.sort_index(level=['site_id', 'datetime'])
        y_val = y_val.sort_index(level=['site_id', 'datetime'])

    X_test = X[datetime_idx >= test_start]
    y_test = y[datetime_idx >= test_start]
    if not X_test.empty:
        X_test = X_test.sort_index(level=['site_id', 'datetime'])
        y_test = y_test.sort_index(level=['site_id', 'datetime'])

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, estimator_params, search_params):
    """
    Performs hyperparameter tuning using RandomizedSearchCV and returns the best model.
    """
    base_model = xgb.XGBRegressor(**estimator_params)

    tscv = TimeSeriesSplit(n_splits=search_params['n_cv_splits'])
    logging.info("Using TimeSeriesSplit with %d splits for cross-validation.", search_params['n_cv_splits'])

    param_dist = _parse_param_distributions(search_params['param_distributions'])

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=search_params['n_iterations'],
        scoring=search_params['scoring'],
        cv=tscv,
        random_state=search_params['random_state'],
        n_jobs=estimator_params.get('n_jobs', -1),
        verbose=search_params['verbose']
    )

    logging.info("Starting RandomizedSearchCV with %d iterations...", search_params['n_iterations'])
    random_search.fit(X_train, y_train)

    logging.info("Best hyperparameters found: %s", random_search.best_params_)
    logging.info("Best cross-validation RMSE: %.4f", -random_search.best_score_)

    best_model = random_search.best_estimator_
    _log_feature_importances(best_model, X_train.columns)

    return best_model, random_search.best_params_


def retrain_with_constraints(best_params, X_train, y_train, constraint_map, base_params):
    """
    Retrains a final XGBoost model using the best hyperparameters and adds
    monotonic constraints.
    """
    logging.info("Defining final model with optimal hyperparameters and monotonic constraints.")

    feature_names = X_train.columns.tolist()
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    monotone_constraints = [0] * len(feature_names)
    for feature, constraint in constraint_map.items():
        if feature in feature_indices:
            monotone_constraints[feature_indices[feature]] = constraint
            logging.info("Applying constraint for '%s': %d", feature, constraint)

    final_params = base_params.copy()
    final_params.update(best_params)
    final_params['monotone_constraints'] = tuple(monotone_constraints)

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_train, y_train)

    return final_model


def evaluate_model(model, X_set, y_set, set_name, target_ids):
    """
    Evaluates the model on a given dataset (validation or test) and logs metrics per site.
    """
    if X_set.empty or y_set.empty:
        logging.warning("%s set is empty. Skipping evaluation.", set_name.capitalize())
        return None

    y_pred = model.predict(X_set)
    results_df = pd.DataFrame({'Actual': y_set, 'Predicted': y_pred}, index=y_set.index)

    available_sites = set(results_df.index.get_level_values('site_id'))
    
    for site_id in target_ids:
        if site_id not in available_sites:
            continue

        site_df = results_df.loc[site_id]
        mae = mean_absolute_error(site_df['Actual'], site_df['Predicted'])
        rmse = np.sqrt(mean_squared_error(site_df['Actual'], site_df['Predicted']))
        r2 = r2_score(site_df['Actual'], site_df['Predicted'])

        logging.info("Metrics for %s SET - %s:", set_name.upper(), site_id)
        logging.info("  MAE: %.3f MW", mae)
        logging.info("  RMSE: %.3f MW", rmse)
        logging.info("  R-squared: %.3f", r2)

    return results_df


def save_model(model, path, filename):
    """Saves trained model using pkl."""
    save_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved successfully to: %s", save_path)


def load_model(path, filename):
    """Loads trained model."""
    load_path = os.path.join(path, filename)
    if not os.path.exists(load_path):
        logging.error("Model file not found at: %s", load_path)
        return None
        
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully from: %s", load_path)
    return model
