import logging

import yaml

from src.data_loader import load_csv_data, load_era5_data, unzip_era5_files
from src.feature_engineering import create_features_for_model
from src.modeling import (
    evaluate_model,
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

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['logging_settings'])

    paths = config['paths']
    data_cfg = config['data_ingestion_params']
    feature_cfg = config['feature_params']
    model_cfg = config['model_params']
    plot_cfg = config['plotting']

    unzip_era5_files(paths['era5_zip_dir'], data_cfg['era5_zip_filenames'], paths['era5_extract_dir'])
    ds_era5 = load_era5_data(paths['era5_extract_dir'], paths['skt_files_path'])
    df_power, df_sites = load_csv_data(paths['power_flow_path'], paths['sites_path'], data_cfg['power_csv_cols'])

    if df_power is None or df_sites is None or ds_era5 is None:
        logging.error("Critical data failed to load. Aborting pipeline.")
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
        logging.error("Preprocessing resulted in an empty DataFrame. Aborting pipeline.")
        return

    X, y = create_features_for_model(
        df=master_df,
        target_col=feature_cfg['target_column'],
        weather_vars=data_cfg['era5_vars'],
        weather_vars_map=data_cfg['weather_vars_map'],
        tcc_var_name=feature_cfg['tcc_var_name']
    )

    if X.empty or y.empty:
        logging.error("Feature engineering resulted in empty X or y. Aborting pipeline.")
        return

    run_quality_assessment(X, y)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, model_cfg['split_dates'])
    sanity_check_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    logging.info("Running hyperparameter search...")
    best_model_from_search, best_params = train_model(X_train, y_train, model_cfg['estimator_params'], model_cfg['hyperparameter_search'])

    logging.info("Training final model with constraints...")
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

    logging.info("Saving final model...")
    save_model(final_model, paths['model_output_dir'], paths['output_model_filename'])

if __name__ == '__main__':
    main()
