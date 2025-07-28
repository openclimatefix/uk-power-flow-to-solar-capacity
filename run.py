"""Main pipeline execution script for the power forecasting project.

This script orchestrates the entire workflow, from data processing to model
training and scenario analysis, based on command-line arguments.
"""
import argparse
import logging
from typing import Any

import pandas as pd
import yaml

from src.data_loader import load_csv_data, load_era5_data, unzip_era5_files
from src.feature_engineering import create_features_for_model
from src.modeling import (
    evaluate_model,
    load_model,
    retrain_with_constraints,
    save_model,
    split_data,
    train_model,
)
from src.preprocessing import create_master_dataframe
from src.scenario_generation_a import run_scenario_analysis as run_overwrite_analysis
from src.scenario_generation_b import run_historical_analogy_analysis
from src.utils import (
    log_and_plot_feature_importance,
    plot_predictions,
    run_quality_assessment,
    sanity_check_splits,
)


def setup_logging(logging_config: dict[str, Any]) -> None:
    """Configure the root logger for the application."""
    logging.basicConfig(
        level=logging_config.get("level", "INFO"),
        format=logging_config.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
        datefmt=logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def run_data_processing(config: dict[str, Any]) -> None:
    """Run the data ingestion, preprocessing, and feature engineering stage."""
    paths = config["paths"]
    data_cfg = config["data_ingestion_params"]
    feature_cfg = config["feature_params"]

    unzip_era5_files(
        paths["era5_zip_dir"], data_cfg.get("era5_zip_filenames", []), paths["era5_extract_dir"],
    )
    ds_era5 = load_era5_data(paths["era5_extract_dir"], paths["skt_files_path"])
    df_power, df_sites = load_csv_data(
        paths["power_flow_path"], paths["sites_path"], data_cfg["power_csv_cols"],
    )

    if df_power is None or df_sites is None or ds_era5 is None:
        logging.error("Critical data failed to load. Aborting.")
        return

    master_df = create_master_dataframe(
        target_ids=data_cfg["target_transformer_ids"],
        df_power=df_power,
        df_sites=df_sites,
        ds_era5=ds_era5,
        start_date=data_cfg["analysis_start_date"],
        end_date=data_cfg["analysis_end_date"],
        era5_vars=data_cfg["era5_vars"],
    )
    ds_era5.close()

    if master_df.empty:
        logging.error("Preprocessing resulted in an empty DataFrame. Aborting.")
        return

    X, y = create_features_for_model(
        df=master_df,
        feature_params=feature_cfg,
        weather_vars=data_cfg["era5_vars"],
        weather_vars_map=data_cfg["weather_vars_map"],
    )

    run_quality_assessment(X, y)

    master_df.to_csv(paths["processed_master_path"])
    X.to_csv(paths["processed_features_path"])
    y.to_csv(paths["processed_target_path"])
    logging.info(
        "Processed data saved to %s, %s, and %s",
        paths["processed_master_path"],
        paths["processed_features_path"],
        paths["processed_target_path"],
    )


def run_training(config: dict[str, Any]) -> None:
    """Run the model training, evaluation, and saving stage."""
    paths = config["paths"]
    data_cfg = config["data_ingestion_params"]
    model_cfg = config["model_params"]
    plot_cfg = config["plotting"]

    X = pd.read_csv(
        paths["processed_features_path"], index_col=["site_id", "datetime"], parse_dates=True,
    )
    y = pd.read_csv(
        paths["processed_target_path"], index_col=["site_id", "datetime"], parse_dates=True,
    ).squeeze()

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, model_cfg["split_dates"])
    sanity_check_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    _, best_params = train_model(
        X_train, y_train, model_cfg["estimator_params"], model_cfg["hyperparameter_search"],
    )

    final_model = retrain_with_constraints(
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        constraint_map=model_cfg.get("monotonic_constraints", {}),
        base_params=model_cfg["estimator_params"],
    )

    log_and_plot_feature_importance(final_model, X_train.columns, plot_cfg)

    val_results = evaluate_model(
        final_model, X_val, y_val, "Validation", data_cfg["target_transformer_ids"],
    )
    test_results = evaluate_model(
        final_model, X_test, y_test, "Test", data_cfg["target_transformer_ids"],
    )

    plot_predictions(val_results, "Validation", data_cfg["target_transformer_ids"], plot_cfg)
    plot_predictions(test_results, "Test", data_cfg["target_transformer_ids"], plot_cfg)

    save_model(final_model, paths["model_output_dir"], paths["xgboost_model_filename"])


def run_inference(config: dict[str, Any]) -> None:
    """Run a simple inference example on a trained model."""
    paths = config["paths"]

    model = load_model(paths["model_output_dir"], paths["xgboost_model_filename"])
    if model is None:
        return

    idx = pd.MultiIndex.from_product(
        [["site_a"], pd.to_datetime(pd.date_range("2025-01-01", periods=24, freq="h"))],
        names=["site_id", "datetime"],
    )

    new_X = pd.DataFrame(index=idx, data={col: range(24) for col in model.feature_names_in_})
    predictions = model.predict(new_X)

    results = pd.DataFrame(
        {"timestamp": new_X.index.get_level_values("datetime"), "predicted_power_mw": predictions},
    )
    logging.info("\n--- Prediction Results ---\n%s", results.to_string())


def run_scenarios(config: dict[str, Any]) -> None:
    """Run the embedded solar capacity estimation stage."""
    paths = config["paths"]
    model_cfg = config["model_params"]
    analysis_method = config["scenario_analysis"]["method"]

    model = load_model(paths["model_output_dir"], paths["xgboost_model_filename"])
    if model is None:
        return

    if analysis_method == "weather_overwrite":
        master_df = pd.read_csv(
            paths["processed_master_path"], index_col="datetime", parse_dates=True,
        )
        config["scenario_analysis_params"] = config["scenario_analysis"]["overwrite_params"]
        run_overwrite_analysis(model=model, master_df=master_df, config=config)

    elif analysis_method == "historical_analogy":
        X = pd.read_csv(
            paths["processed_features_path"], index_col=["site_id", "datetime"], parse_dates=True,
        )
        y = pd.read_csv(
            paths["processed_target_path"], index_col=["site_id", "datetime"], parse_dates=True,
        ).squeeze()
        X_train, _, _, _, X_test, y_test = split_data(X, y, model_cfg["split_dates"])
        config["historical_analogy_params"] = config["scenario_analysis"]["analogy_params"]
        run_historical_analogy_analysis(
            model=model, X_historical=X_train, X_test=X_test, y_test=y_test, config=config,
        )
    else:
        logging.error("Unknown analysis method in config: '%s'", analysis_method)


def main() -> None:
    """Parse command-line arguments and run the selected pipeline stage."""
    parser = argparse.ArgumentParser(description="Run stages of the power forecasting pipeline.")
    parser.add_argument(
        "stage",
        choices=["process-data", "train", "run-scenarios", "full-pipeline", "predict"],
        help="The stage of the pipeline to run.",
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    setup_logging(config["logging_settings"])

    if args.stage == "process-data":
        run_data_processing(config)
    elif args.stage == "train":
        run_training(config)
    elif args.stage == "run-scenarios":
        run_scenarios(config)
    elif args.stage == "predict":
        run_inference(config)
    elif args.stage == "full-pipeline":
        run_data_processing(config)
        run_training(config)
        run_scenarios(config)


if __name__ == "__main__":
    main()
