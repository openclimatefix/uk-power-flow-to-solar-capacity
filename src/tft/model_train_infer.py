
import gc
import json
import logging
import warnings
import yaml
from typing import Dict, Any

import torch
import lightning as L
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, SMAPE

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
gc.collect()


def load_best_hyperparameters(results_path: str) -> Dict[str, Any]:
    logger.info(f"Loading best hyperparameters from: {results_path}")
    with open(results_path, 'r') as f:
        optuna_results = json.load(f)
    
    optimal_config = optuna_results['best_params']
    logger.info("Best parameters from Optuna:")
    for key, value in optimal_config.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return optimal_config


def save_best_config(optimal_config: Dict[str, Any], save_path: str = 'best_tft_config.json'):
    with open(save_path, 'w') as f:
        json.dump(optimal_config, f, indent=2)
    logger.info(f"Configuration saved to: {save_path}")


def create_final_model(training_dataset, optimal_config: Dict[str, Any]):
    # TFT INSTANTIATION WITH BEST PARAMS
    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=optimal_config["learning_rate"],
        hidden_size=optimal_config["hidden_size"], 
        attention_head_size=optimal_config["attention_head_size"],
        dropout=optimal_config["dropout"],
        reduce_on_plateau_patience=6,
        optimizer="adam"
    )
    
    return tft_model


def train_final_model(model, train_dataloader, val_dataloader, optimal_config: Dict[str, Any]):
    # TRAINER DEFINE - x4 EPOCHS AFTER
    trainer = L.Trainer(
        max_epochs=25,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=optimal_config["gradient_clip_val"],
        enable_progress_bar=True,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                mode="min",
                verbose=True
            )
        ]
    )
    
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
    checkpoint_path = "production_tft_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    logger.info(f"Lightning checkpoint saved to: {checkpoint_path}")
    
    # SAVE TRAINED MODEL
    metadata_save_path = "production_tft_metadata.pth"
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
            'target': training_dataset.target
        }
    }, metadata_save_path)
    
    logger.info(f"Model metadata saved to: {metadata_save_path}")


def test_prediction(model, val_dataloader):
    # INFER FUNCTION DEFINE - BASE
    model.eval()
    
    with torch.no_grad():
        predictions = model.predict(val_dataloader, mode="prediction", return_x=True)
    
    if hasattr(predictions, 'prediction'):
        pred_tensor = predictions.prediction
        logger.info(f"Predictions shape: {pred_tensor.shape}")
        logger.info(f"Prediction range: {pred_tensor.min():.3f} to {pred_tensor.max():.3f}")
    
    return True
