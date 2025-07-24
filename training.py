import yaml
import logging
import pandas as pd

from src.data_loader import unzip_era5_files, load_era5_data, load_csv_data
from src.preprocessing import create_master_dataframe
from src.feature_engineering import create_features_for_model
from src.modeling import split_data, train_model, retrain_with_constraints, evaluate_model, save_model
from src.utils import sanity_check_splits, log_and_plot_feature_importance, plot_predictions


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

    idx = pd.MultiIndex.from_product(
        [data_cfg['target_transformer_ids'], pd.to_datetime(pd.date_range('2021-01-01', '2024-12-31', freq='h', tz='UTC'))],
        names=['site_id', 'datetime']
    )
    dummy_X = pd.DataFrame(index=idx, data={'tcc_lag_1h': range(len(idx)), 't2m_lag_1h': range(len(idx)), 'ssrd_lag_1h': range(len(idx))})
    dummy_y = pd.Series(index=idx, data=range(len(idx)), name='power')
    X, y = dummy_X, dummy_y

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

    save_model(final_model, paths['model_output_dir'], paths['output_model_filename'])


if __name__ == '__main__':
    main()
