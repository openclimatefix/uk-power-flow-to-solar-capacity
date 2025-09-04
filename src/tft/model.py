import json
import logging
import os
import warnings
from typing import Any

import dask.dataframe as dd
import lightning as L
import pandas as pd
import pytorch_forecasting as ptf
import torch
import yaml
import glob

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import gc

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
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
        try:
            df = dd.read_parquet(
                self.dataset_path, 
                index=False,
                schema="infer",
                split_row_groups=False,
                ignore_metadata_file=True,
            )
        except Exception as e:
            logger.warning(f"First attempt failed: {e}")
            parquet_files = glob.glob(f"{self.dataset_path}/*.parquet")
            logger.info(f"Found {len(parquet_files)} parquet files")
            
            dfs = []
            for file in parquet_files:
                logger.info(f"Reading {file}")
                df_part = dd.read_parquet(file, index=False)
                dfs.append(df_part)
            
            df = dd.concat(dfs, ignore_index=True)
        
        logger.info(f"Data types distribution:\n{df.dtypes.value_counts()}")
        return df

    def check_nan_values(self, ddf: dd.DataFrame) -> dict[str, int]:
        try:
            nan_counts = ddf.isnull().sum().compute()
        except Exception as e:
            logger.warning(f"NaN check failed with dask - converting to pandas: {e}")
            df_pandas = ddf.compute()
            nan_counts = df_pandas.isnull().sum()
        
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
        
        if lag_cols:
            df_clean = df.dropna(subset=lag_cols)
            logger.info("Computing cleaned dataset length...")
            cleaned_length = len(df_clean)
            logger.info(f"Cleaned dataset: {cleaned_length:,} rows")
        else:
            df_clean = df
            logger.info("No lag columns found, skipping dropna")

        columns_to_drop = config["columns_to_drop"]
        existing_cols = [col for col in columns_to_drop if col in df_clean.columns]
        if existing_cols:
            df_clean = df_clean.drop(existing_cols, axis=1)
            logger.info(f"Dropped columns: {existing_cols}")
        
        try:
            remaining_nans = df_clean.isnull().sum().sum().compute()
            logger.info(f"Remaining NaNs: {remaining_nans}")
        except Exception as e:
            logger.warning(f"Could not compute remaining NaNs: {e}")

        return df_clean

    def prepare_for_tft(self, df: dd.DataFrame, output_path: str) -> pd.DataFrame:
        df_pandas = df.compute()        
        df_pandas = df_pandas.sort_values(["location", "timestamp"])
        df_pandas["time_idx"] = df_pandas.groupby("location").cumcount()
        df_pandas["location"] = df_pandas["location"].astype(str)        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_pandas.to_parquet(output_path, index=False)
        return df_pandas


def build_search_space(spec: dict) -> dict:
    """Builds a Ray Tune search space from a dictionary specification."""
    search_space = {}
    for param, settings in spec.items():
        search_type = settings.pop("type")
        if search_type == "choice":
            search_space[param] = tune.choice(settings["values"])
        elif search_type == "uniform":
            search_space[param] = tune.uniform(settings["low"], settings["high"])
        elif search_type == "loguniform":
            search_space[param] = tune.loguniform(settings["low"], settings["high"])
        else:
            raise ValueError(f"Unsupported search space type: {search_type}")
    return search_space


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
        num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() > 2 else 0
        
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, 
            batch_size=config["training"]["batch_size"], 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=torch.cuda.is_available()
        )
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, 
            batch_size=config["training"]["batch_size"], 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=torch.cuda.is_available()
        )
        test_dataloader = self.test_dataset.to_dataloader(
            train=False, 
            batch_size=config["training"]["batch_size"], 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=torch.cuda.is_available()
        )
        return train_dataloader, val_dataloader, test_dataloader

    def optimize_with_ray(self):
        try:
            ray.init(ignore_reinit_error=True)            
            
            ray_config = config["ray"]
            ray_data = ray.put(self.df)
            num_locations = self.num_locations
            
            def ray_objective_func(config_params):
                torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    df = ray.get(ray_data)
                    locations = df["location"].unique()[:num_locations]
                    df_multi = df[df["location"].isin(locations)].copy()
                    max_time_idx = df_multi["time_idx"].max()
                    cv_end = int(max_time_idx * 0.8)
                    split_size = cv_end // 4
                    
                    cv_scores = []
                    
                    for fold in range(2):
                        train_end = (fold + 2) * split_size
                        val_start = train_end + 1
                        val_end = min(val_start + split_size, cv_end)                
                        train_data = df_multi[df_multi["time_idx"] <= train_end]
                        val_data = df_multi[(df_multi["time_idx"] > train_end) & (df_multi["time_idx"] <= val_end)]
                        if len(val_data) == 0:
                            continue
                            
                        fold_train_dataset = TimeSeriesDataSet(
                            train_data,
                            time_idx=TFTConfig.TIME_IDX, target=TFTConfig.TARGET,
                            group_ids=TFTConfig.GROUP_IDS, max_encoder_length=TFTConfig.MAX_ENCODER_LENGTH,
                            max_prediction_length=TFTConfig.MAX_PREDICTION_LENGTH,
                            static_categoricals=TFTConfig.STATIC_CATEGORICALS,
                            static_reals=TFTConfig.STATIC_REALS,
                            time_varying_known_reals=TFTConfig.TIME_VARYING_KNOWN_REALS,
                            time_varying_unknown_reals=TFTConfig.TIME_VARYING_UNKNOWN_REALS,
                            add_relative_time_idx=True, add_target_scales=True,
                            add_encoder_length=True,
                        )
                        
                        fold_val_dataset = TimeSeriesDataSet.from_dataset(
                            fold_train_dataset, val_data, predict=False, stop_randomization=True,
                        )
                        
                        # Pass config_params dict directly to the model
                        model = TemporalFusionTransformer.from_dataset(fold_train_dataset, **config_params)
                        # Add a non-tunable param back in
                        model.hparams.reduce_on_plateau_patience = 6

                        trainer = L.Trainer(
                            max_epochs=10, 
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices=1, 
                            gradient_clip_val=config_params["gradient_clip_val"],
                            enable_progress_bar=False, logger=False, enable_checkpointing=False,
                            callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")],
                        )

                        num_workers = min(8, os.cpu_count() // 4)
                        fold_train_dl = fold_train_dataset.to_dataloader(train=True, batch_size=256, num_workers=num_workers)
                        fold_val_dl = fold_val_dataset.to_dataloader(train=False, batch_size=256, num_workers=num_workers)
                        
                        trainer.fit(model, fold_train_dl, fold_val_dl)
                        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
                        cv_scores.append(float(val_loss))                
                        del model, trainer, fold_train_dataset, fold_val_dataset, fold_train_dl, fold_val_dl
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    mean_cv_score = sum(cv_scores) / len(cv_scores) if cv_scores else float("inf")
                    
                    from ray.air import session
                    session.report({"val_loss": mean_cv_score, "cv_scores": cv_scores})

                except Exception as e:
                    logger.error(f"Trial failed with error: {e}")
                    from ray.air import session
                    session.report({"val_loss": float("inf")})
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            search_space = build_search_space(ray_config["search_space"])
            scheduler_config = ray_config["scheduler"]
            
            scheduler = ASHAScheduler(
                metric=scheduler_config["metric"], 
                mode=scheduler_config["mode"], 
                max_t=scheduler_config["max_t"], 
                grace_period=scheduler_config["grace_period"],
                reduction_factor=scheduler_config["reduction_factor"],
            )
            
            analysis = tune.run(
                ray_objective_func,
                config=search_space,
                num_samples=ray_config["n_trials"],
                scheduler=scheduler,
                resources_per_trial={"cpu": 8, "gpu": 1.0},
                max_concurrent_trials=ray_config.get("max_concurrent_trials", 1),
                max_failures=3,
                raise_on_failed_trial=False,
                storage_path="/tmp/ray_results",
                name="tft_optimization",
                verbose=1,
                resume=True,
            )
            
            best_trial = analysis.get_best_trial("val_loss", "min", "last")
            if best_trial:
                return {
                    "best_params": best_trial.config, 
                    "best_value": best_trial.last_result["val_loss"],
                    "n_trials": len(analysis.trials)
                }
            else:
                logger.error("No successful trials completed.")
                return {"best_params": {}, "best_value": float("inf"), "error": "No successful trials."}

        finally:
            ray.shutdown()


def main():
    processor = DataProcessor(config["paths"]["dataset_path"])

    # POST PROCESS PRE DATALODERS
    df = processor.load_data()
    processor.check_nan_values(df)
    df_clean = processor.clean_data(df)
    df_pandas = processor.prepare_for_tft(df_clean, config["paths"]["output_path"])

    # INIT MODEL - DATALOADERS / CROSS VAL - RAY TRIALS
    trainer = TFTTrainer(df_pandas)
    trainer.create_datasets()
    optimization_results = trainer.optimize_with_ray()

    # SAVE BEST HPARAMS
    with open(config["paths"]["results_path"], "w") as f:
        json.dump(optimization_results, f, indent=2)
    logger.info(f"Hyperparameters and configuration saved to: {config['paths']['results_path']}")


if __name__ == "__main__":
    main()
