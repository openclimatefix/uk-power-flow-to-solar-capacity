from __future__ import annotations

import logging

from model import TFTTrainer
from utils import (
    configure_logging,
    get_logger,
    load_yaml,
    ensure_keys,
    restore_ray_best_checkpoint,
    load_tft_from_checkpoint,
    load_parquet,
    fmt_metric,
)
from plotting import (
    create_interpretation_plot_from_model,
    create_correlation_plot,
)


logger = configure_logging(level="INFO")
logger = get_logger("rae")


try:
    CONFIG_PATH = "uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml"
    config = load_yaml(CONFIG_PATH)

    ensure_keys(config, ["analysis"])
    ensure_keys(config, ["analysis", "experiment_path"])
    ensure_keys(config, ["analysis", "metrics"])
    ensure_keys(config, ["analysis", "metrics", "primary_metric"])
    ensure_keys(config, ["analysis", "metrics", "mode"])
    ensure_keys(config, ["analysis", "data_path"])

    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise


def load_best_model():
    """
    Restore the Ray Tune experiment, pick the best checkpoint by the configured metric,
    and load the TFT model from that checkpoint.
    """
    experiment_path = config["analysis"]["experiment_path"]
    logger.info(f"Loading results from: {experiment_path}")

    try:
        metrics_config = config["analysis"]["metrics"]
        metric = metrics_config["primary_metric"]
        mode = metrics_config["mode"]

        best_result, best_checkpoint = restore_ray_best_checkpoint(
            experiment_path=experiment_path,
            trainable="ray_objective_func",
            metric=metric,
            mode=mode,
        )

        val_loss = best_result.metrics[metric]
        logger.info(f"Best trial's final validation loss: {fmt_metric(val_loss)}")

        best_tft_model = load_tft_from_checkpoint(best_checkpoint.path)
        logger.info("Best model loaded successfully")

        return best_tft_model, float(val_loss)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_data_and_setup_trainer():
    """
    Load the dataset and construct the validation dataloader via TFTTrainer.
    """
    data_path = config["analysis"]["data_path"]
    logger.info(f"Loading data from: {data_path}")

    try:
        df_pandas = load_parquet(data_path)
        logger.info(f"Loaded dataset with {len(df_pandas):,} rows")

        trainer_setup = TFTTrainer(df_pandas)
        trainer_setup.create_datasets()
        _, val_dataloader, _ = trainer_setup.create_dataloaders()
        logger.info("Validation dataloader created")

        return df_pandas, val_dataloader

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def analyse_experiment_results():
    logger.info("Starting TFT experiment analysis...")

    try:
        best_model, val_loss = load_best_model()
        df_pandas, val_dataloader = load_data_and_setup_trainer()

        create_interpretation_plot_from_model(best_model, val_dataloader, config)

        feature_columns = best_model.hparams.time_varying_known_reals
        create_correlation_plot(df_pandas, feature_columns, config)

        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Best validation loss: {fmt_metric(val_loss)}")
        logger.info(f"Features analysed: {len(feature_columns)}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyse_experiment_results()
