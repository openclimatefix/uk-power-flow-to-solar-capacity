import json
import logging
import os
import warnings
from typing import Any

import dask.dataframe as dd
import lightning as L
import optuna
import pandas as pd
import pytorch_forecasting as ptf
import torch
import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

with open("config.yaml") as f:
    config = yaml.safe_load(f)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)


handlers = []
if config["logging"]["file"]:
    handlers.append(logging.FileHandler(config["paths"]["log_file"]))
if config["logging"]["console"]:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"],
    handlers=handlers,
)
logger = logging.getLogger(__name__)


class TFTConfig:
    """Configuration class for TFT model params"""

    STATIC_CATEGORICALS = config["model"]["static_categoricals"]
    STATIC_REALS = config["model"]["static_reals"]
    TIME_VARYING_KNOWN_REALS = config["model"]["time_varying_known_reals"]
    TIME_VARYING_UNKNOWN_REALS = config["model"]["time_varying_unknown_reals"]
    TARGET = config["model"]["target"]
    TIME_IDX = config["model"]["time_idx"]
    GROUP_IDS = config["model"]["group_ids"]
    MAX_ENCODER_LENGTH = config["model"]["max_encoder_length"]
    MAX_PREDICTION_LENGTH = config["model"]["max_prediction_length"]


class DataProcessor:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_data(self) -> dd.DataFrame:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        df = dd.read_parquet(self.dataset_path)
        logger.info(f"Dataset loaded successfully: {len(df.columns)} columns")
        logger.info(f"Data types distribution:\n{df.dtypes.value_counts()}")
        return df

    def check_nan_values(self, ddf: dd.DataFrame) -> dict[str, int]:

        nan_counts = ddf.isnull().sum().compute()
        total_nans = {col: count for col, count in nan_counts.items() if count > 0}

        if total_nans:
            logger.info("Columns with missing values:")
            for col, count in sorted(total_nans.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {col}: {count:,} NaNs")
        else:
            logger.info("No NaN values found!")

        return total_nans

    def clean_data(self, df: dd.DataFrame) -> dd.DataFrame:

        lag_cols = [col for col in df.columns if "_lag_" in col]
        df_clean = df.dropna(subset=lag_cols)
        logger.info(f"Cleaned dataset: {len(df_clean):,} rows")

        columns_to_drop = config["columns_to_drop"]
        existing_cols = [col for col in columns_to_drop if col in df_clean.columns]
        if existing_cols:
            df_clean = df_clean.drop(existing_cols, axis=1)
            logger.info(f"Dropped columns: {existing_cols}")

        logger.info(f"Final dataset: {len(df_clean.columns)} columns")

        remaining_nans = df_clean.isnull().sum().sum().compute()
        logger.info(f"Remaining NaNs: {remaining_nans}")

        return df_clean

    def prepare_for_tft(self, df: dd.DataFrame, output_path: str) -> pd.DataFrame:

        df_pandas = df.compute()
        df_pandas = df_pandas.sort_values(["location", "timestamp"])
        df_pandas["time_idx"] = df_pandas.groupby("location").cumcount()
        df_pandas["location"] = df_pandas["location"].astype(str)
        df_pandas.to_parquet(output_path, index=False)
        logger.info(f"TFT-ready data saved to: {output_path}")

        return df_pandas


class TFTTrainer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_locations = config["num_locations"]
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def create_datasets(self):
        """Create training, validation and test datasets"""
        locations = self.df["location"].unique()[:self.num_locations]
        df_multi = self.df[self.df["location"].isin(locations)].copy()
        logger.info(f"Using {len(locations)} locations: {list(locations)}")
        logger.info(f"Dataset size: {len(df_multi):,} rows")

        max_time_idx = df_multi["time_idx"].max()
        train_cutoff = int(max_time_idx * config["train_split"])
        val_cutoff = int(max_time_idx * config["val_split"])

        logger.info(f"Train: 0 to {train_cutoff} (2021-2023)")
        logger.info(f"Validation: {train_cutoff} to {val_cutoff} (2024)")
        logger.info(f"Test: {val_cutoff} to {max_time_idx} (2025)")

        # TRAINING DATASET
        self.training_dataset = TimeSeriesDataSet(
            df_multi[df_multi["time_idx"] <= train_cutoff],
            time_idx=TFTConfig.TIME_IDX,
            target=TFTConfig.TARGET,
            group_ids=TFTConfig.GROUP_IDS,
            max_encoder_length=TFTConfig.MAX_ENCODER_LENGTH,
            max_prediction_length=TFTConfig.MAX_PREDICTION_LENGTH,
            static_categoricals=TFTConfig.STATIC_CATEGORICALS,
            static_reals=TFTConfig.STATIC_REALS,
            time_varying_known_reals=TFTConfig.TIME_VARYING_KNOWN_REALS,
            time_varying_unknown_reals=TFTConfig.TIME_VARYING_UNKNOWN_REALS,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        logger.info(f"Training samples: {len(self.training_dataset)}")

        # VALIDATION DATASET
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df_multi[
                (df_multi["time_idx"] > train_cutoff) &
                (df_multi["time_idx"] <= val_cutoff)
            ],
            predict=False,
            stop_randomization=True,
        )

        logger.info(f"Validation samples: {len(self.validation_dataset)}")

        # TEST DATASET
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df_multi[df_multi["time_idx"] > val_cutoff],
            predict=False,
            stop_randomization=True,
        )

        logger.info(f"Test samples: {len(self.test_dataset)}")

    def create_dataloaders(self):
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=config["training"]["batch_size"], num_workers=0,
        )
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=config["training"]["batch_size"], num_workers=0,
        )
        test_dataloader = self.test_dataset.to_dataloader(
            train=False, batch_size=config["training"]["batch_size"], num_workers=0,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def objective(self, trial: optuna.Trial) -> float:
        # OPTUNA OBJECTIVE FUNCTION - RAY FOR PRODUCTION
        torch.cuda.empty_cache()

        search_space = config["optuna"]["search_space"]
        opt_config = {}

        for param, space in search_space.items():
            if space["type"] == "loguniform":
                opt_config[param] = trial.suggest_loguniform(param, space["low"], space["high"])
            elif space["type"] == "categorical":
                opt_config[param] = trial.suggest_categorical(param, space["choices"])
            elif space["type"] == "uniform":
                opt_config[param] = trial.suggest_uniform(param, space["low"], space["high"])

        try:
            model = TemporalFusionTransformer.from_dataset(
                self.training_dataset,
                learning_rate=opt_config["learning_rate"],
                hidden_size=opt_config["hidden_size"],
                attention_head_size=opt_config["attention_head_size"],
                dropout=opt_config["dropout"],
                reduce_on_plateau_patience=config["training"]["reduce_on_plateau_patience"],
            )

            trainer = L.Trainer(
                max_epochs=config["training"]["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                gradient_clip_val=opt_config["gradient_clip_val"],
                enable_progress_bar=config["training"]["enable_progress_bar"],
                logger=False,
                enable_checkpointing=config["training"]["enable_checkpointing"],
                callbacks=[L.pytorch.callbacks.EarlyStopping(
                    monitor="val_loss", patience=config["training"]["early_stopping_patience"], mode="min",
                )],
            )

            train_dataloader, val_dataloader, _ = self.create_dataloaders()
            trainer.fit(model, train_dataloader, val_dataloader)
            val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
            return float(val_loss)

        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return float("inf")
        finally:
            torch.cuda.empty_cache()

    def optimize_hyperparameters(self) -> dict[str, Any]:
        optuna_config = config["optuna"]
        logger.info(f"Starting hyperparameter optimization with {optuna_config['n_trials']} trials...")
        logger.info(f"PyTorch Lightning: {L.__version__}")
        logger.info(f"PyTorch Forecasting: {ptf.__version__}")

        torch.cuda.empty_cache()
        study = optuna.create_study(direction=optuna_config["direction"])
        study.optimize(self.objective, n_trials=optuna_config["n_trials"], timeout=optuna_config["timeout"])

        # UPDATE TO SAVE AS JSON - REFER TO NEXT BLOCK
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")

        results = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "optimization_complete": True,
            "model_config": {
                "max_encoder_length": TFTConfig.MAX_ENCODER_LENGTH,
                "max_prediction_length": TFTConfig.MAX_PREDICTION_LENGTH,
                "target": TFTConfig.TARGET,
                "time_idx": TFTConfig.TIME_IDX,
                "group_ids": TFTConfig.GROUP_IDS,
                "static_categoricals": TFTConfig.STATIC_CATEGORICALS,
                "time_varying_known_reals": TFTConfig.TIME_VARYING_KNOWN_REALS,
            },
        }

        return {
            "results": results,
            "study": study,
        }


def main():
    """Main execution function"""
    processor = DataProcessor(config["paths"]["dataset_path"])

    df = processor.load_data()
    processor.check_nan_values(df)
    df_clean = processor.clean_data(df)
    df_pandas = processor.prepare_for_tft(df_clean, config["paths"]["output_path"])

    trainer = TFTTrainer(df_pandas)
    trainer.create_datasets()

    optimization_results = trainer.optimize_hyperparameters()

    # SAVE BEST HPARAMS VIA OPTUNA OBJECTIVE (NOT the model)
    with open(config["paths"]["results_path"], "w") as f:
        json.dump(optimization_results["results"], f, indent=2)
    logger.info(f"Hyperparameters and configuration saved to: {config['paths']['results_path']}")


if __name__ == "__main__":
    main()
