import gc
import os
import json
import logging
import warnings
import yaml
from typing import Dict, Any

import torch
import lightning as L
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
    config = yaml.safe_load(f)


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)


handlers = []
if config['logging']['file']:
    handlers.append(logging.FileHandler('tft_final_training.log'))
if config['logging']['console']:
    handlers.append(logging.StreamHandler())


logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format=config['logging']['format'],
    handlers=handlers
)


logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
gc.collect()


def create_final_model(training_dataset, model_params: Dict[str, Any]):
    """Instantiates the TFT model with architecture-specific hyperparameters."""
    logger.info("Instantiating model with optimal hyperparameters...")

    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        **model_params,
        loss=QuantileLoss(),
    )

    tft_model.hparams.learning_rate = model_params["learning_rate"]

    return tft_model


def train_final_model(model, train_dataloader, val_dataloader, trainer_params: Dict[str, Any]):
    logger.info("Configuring trainer...")
    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        gradient_clip_val=trainer_params.get("gradient_clip_val", 1.0),
        enable_progress_bar=True,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config['training']['early_stopping_patience'],
                mode="min",
                verbose=True
            )
        ]
    )
    
    logger.info("Starting model fitting...")
    trainer.fit(model, train_dataloader, val_dataloader)    
    
    final_val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
    final_train_loss = trainer.callback_metrics.get('train_loss', 'N/A')
    epochs_completed = trainer.current_epoch + 1
    
    logger.info(f"Final validation loss: {final_val_loss}")
    logger.info(f"Final train loss: {final_train_loss}")
    logger.info(f"Epochs completed: {epochs_completed}")
    
    return trainer, final_val_loss, final_train_loss, epochs_completed


def save_model(trainer, model, optimal_config: Dict[str, Any], 
               final_val_loss, final_train_loss, epochs_completed, 
               training_dataset):
    """Saves the model checkpoint and a separate metadata file."""
    checkpoint_path = "second_tft_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    logger.info(f"Lightning checkpoint saved to: {checkpoint_path}")
    
    metadata_save_path = "second_tft_metadata.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': optimal_config,
        'final_metrics': {
            'val_loss': float(final_val_loss) if final_val_loss != 'N/A' else None,
            'train_loss': float(final_train_loss) if final_train_loss != 'N/A' else None,
            'epochs': epochs_completed
        },
        'training_config': {
            'max_encoder_length': training_dataset.max_encoder_length,
            'max_prediction_length': training_dataset.max_prediction_length,
            'target': training_dataset.target,
            'group_ids': training_dataset.group_ids
        }
    }, metadata_save_path)
    
    logger.info(f"Model metadata and state dict saved to: {metadata_save_path}")


if __name__ == "__main__":

    import numpy as np

    logger.info(f"Loading cached processed data from {config['paths']['output_path']}")
    try:
        df_pandas_full = pd.read_parquet(config['paths']['output_path'])
        logger.info("Successfully loaded cached data.")
    except FileNotFoundError:
        logger.error(f"Error: The processed data file was not found at {config['paths']['output_path']}")
        logger.error("Please run your main optimization script first to process and cache the data.")
        exit()

    all_locations = df_pandas_full['location'].unique()
    np.random.seed(42)
    selected_locations = np.random.choice(all_locations, size=5, replace=False)
    logger.info(f"Randomly selected 5 sites for training: {selected_locations}")
    df_pandas = df_pandas_full[df_pandas_full['location'].isin(selected_locations)].copy()
    logger.info(f"Filtered dataset to {len(df_pandas)} rows.")

    model_config = config['model']

    logger.info("Creating final training and validation datasets...")
    max_time_idx = df_pandas[model_config['time_idx']].max()
    
    training_cutoff = int(max_time_idx * config["train_split"])
    validation_cutoff = int(max_time_idx * config["val_split"])

    training_dataset = TimeSeriesDataSet(
        df_pandas[df_pandas["time_idx"] <= validation_cutoff],
        time_idx=model_config['time_idx'],
        target=model_config['target'],
        group_ids=model_config['group_ids'],
        max_encoder_length=model_config['max_encoder_length'],
        max_prediction_length=model_config['max_prediction_length'],
        static_categoricals=model_config['static_categoricals'],
        static_reals=model_config['static_reals'],
        time_varying_known_reals=model_config['time_varying_known_reals'],
        time_varying_unknown_reals=model_config['time_varying_unknown_reals'],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df_pandas, predict=False, stop_randomization=True
    )

    train_dl = training_dataset.to_dataloader(train=True, batch_size=config['training']['batch_size'], num_workers=4)
    val_dl = validation_dataset.to_dataloader(train=False, batch_size=config['training']['batch_size'], num_workers=4)
    logger.info("Dataloaders created.")

    # HARDCODED FROM INITIAL RAY TRIALS (FULL FEATURES) - UPDATE ACCORDINGLY
    optimal_hyperparameters = {
        "learning_rate": 0.0001,
        "hidden_size": 16,
        "attention_head_size": 4,
        "dropout": 0.0767335,
        "gradient_clip_val": 0.744715,
        "hidden_continuous_size": 8,
        "lstm_layers": 2,
        "weight_decay": 9.56182e-05
    }

    logger.info(f"Using parameters: {optimal_hyperparameters}")
    model_params = optimal_hyperparameters.copy()
    model_params.pop("gradient_clip_val", None)
    final_model = create_final_model(training_dataset, model_params)

    trainer, val_loss, train_loss, epochs = train_final_model(
        final_model, train_dl, val_dl, optimal_hyperparameters
    )
    
    save_model(
        trainer, final_model, optimal_hyperparameters, 
        val_loss, train_loss, epochs, training_dataset
    )
