# Entire pipeline execution script

import argparse
import logging

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
from src.utils import (
    log_and_plot_feature_importance,
    plot_predictions,
    run_quality_assessment,
    sanity_check_splits,
)


def setup_logging(logging_config):
    logging.basicConfig(
        level=logging_config.get('level', 'INFO'),
        format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        datefmt=logging_config.get('datefmt', '%Y-%m-%d %H:%M:%S')
    )


def run_data_processing(config):
    paths = config['paths']
    data_cfg = config['data_ingestion_params']
    feature_cfg = config['feature_params']

    unzip_era5_files(paths['era5_zip_dir'], data_cfg['era5_zip_filenames'], paths['era5_extract_dir'])
    ds_era5 = load_era5_data(paths['era5_extract_dir'], paths['skt_files_path'])
    df_power, df_sites = load_csv_data(paths['power_flow_path'], paths['sites_path'], data_cfg['power_csv_cols'])

    if df_power is None or df_sites is None or ds_era5 is None:
        logging.error("Critical data failed to load. Aborting.")
        return

    master_df = create_master_dataframe(
        target_ids=data_cfg['target_transformer_ids'],
        df_power=df_power,
        df_sites=df_sites,
        ds_era5=ds_era5,
        start_date=data_cfg['analysis_start_date'],
        end_date=data_cfg['analysis_end_date'],
        era5_vars=data_cfg['era5_vars']
    )
    ds_era5.close()

    if master_df.empty:
        logging.error("Preprocessing resulted in an empty DataFrame. Aborting.")
        return

    X, y = create_features_for_model(
        df=master_df,
        target_col=feature_cfg['target_column'],
        weather_vars=data_cfg['era5_vars'],
        weather_vars_map=data_cfg['weather_vars_map'],
        tcc_var_name=feature_cfg['tcc_var_name']
    )

    run_quality_assessment(X, y)

    # Save processed data
    X.to_csv(paths['processed_features_path'])
    y.to_csv(paths['processed_target_path'])
    logging.info("Processed data saved to %s and %s", paths['processed_features_path'], paths['processed_target_path'])


def run_training(config):
    logging.info("--- STAGE: MODEL TRAINING STARTED ---")
    paths = config['paths']
    data_cfg = config['data_ingestion_params']
    model_cfg = config['model_params']
    plot_cfg = config['plotting']

    # Load processed data
    X = pd.read_csv(paths['processed_features_path'], index_col=['site_id', 'datetime'], parse_dates=True)
    y = pd.read_csv(paths['processed_target_path'], index_col=['site_id', 'datetime'], parse_dates=True).squeeze()

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, model_cfg['split_dates'])
    sanity_check_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    best_model_from_search, best_params = train_model(X_train, y_train, model_cfg['estimator_params'], model_cfg['hyperparameter_search'])

    final_model = retrain_with_constraints(
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        constraint_map=model_cfg.get('monotonic_constraints', {}),
        base_params=model_cfg['estimator_params']
    )

    log_and_plot_feature_importance(final_model, X_train.columns, plot_cfg)

    val_results = evaluate_model(final_model, X_val, y_val, "Validation", data_cfg['target_transformer_ids'])
    test_results = evaluate_model(final_model, X_test, y_test, "Test", data_cfg['target_transformer_ids'])

    plot_predictions(val_results, "Validation", data_cfg['target_transformer_ids'], plot_cfg)
    plot_predictions(test_results, "Test", data_cfg['target_transformer_ids'], plot_cfg)

    save_model(final_model, paths['model_output_dir'], paths['output_model_filename'])


def run_inference(config):
    paths = config['paths']

    model = load_model(paths['model_output_dir'], paths['output_model_filename'])
    if model is None:
        return

    idx = pd.MultiIndex.from_product(
        [['site_a'], pd.to_datetime(pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC'))],
        names=['site_id', 'datetime']
    )

    new_X = pd.DataFrame(index=idx, data={col: range(24) for col in model.feature_names_in_})
    predictions = model.predict(new_X)

    results = pd.DataFrame({'timestamp': new_X.index.get_level_values('datetime'), 'predicted_power_mw': predictions})
    print("\n--- Prediction Results ---")
    print(results)


def main():
    parser = argparse.ArgumentParser(description="Run stages of the power forecasting pipeline.")
    parser.add_argument(
        "stage",
        choices=["process-data", "train", "predict", "full-pipeline"],
        help="The stage of the pipeline to run."
    )
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    setup_logging(config['logging_settings'])

    if args.stage == "process-data":
        run_data_processing(config)
    elif args.stage == "train":
        run_training(config)
    elif args.stage == "predict":
        run_inference(config)
    elif args.stage == "full-pipeline":
        run_data_processing(config)
        run_training(config)


if __name__ == '__main__':
    main()
