"""Full training pipeline for the TFT production model."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from src.tft.data import create_production_datasets
from src.tft.model import create_final_model
from src.tft.utils import find_resume_checkpoint, save_production_artifacts

logger = logging.getLogger(__name__)


def build_callbacks(training_cfg: DictConfig) -> list[pl.Callback]:
    """Build Lightning callback list from config.

    Args:
        training_cfg: cfg.training sub-config.

    Returns:
        List of instantiated callbacks.
    """
    callbacks: list[pl.Callback] = [
        EarlyStopping(
            monitor="val_loss",
            patience=training_cfg.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        RichModelSummary(max_depth=2),
    ]

    if training_cfg.get("enable_checkpointing", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )

    return callbacks


def build_loggers(training_cfg: DictConfig) -> list:
    """Build Lightning logger list from config.

    Args:
        training_cfg: cfg.training sub-config.

    Returns:
        List of instantiated loggers.
    """
    if training_cfg.get("logger") == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return [
            WandbLogger(
                project=training_cfg.get("wandb_project", "tft"),
                log_model=True,
            )
        ]

    return [
        TensorBoardLogger(save_dir="logs", name="tft_tb"),
        CSVLogger(save_dir="logs", name="tft_csv"),
    ]


def train_final_model(
    cfg: DictConfig,
    model: pl.LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    resume_ckpt: str | None = None,
) -> tuple[Trainer, float, float, int]:
    """Fit model and return trainer plus summary metrics.

    Args:
        cfg: Full Hydra config.
        model: Instantiated TFTWithGRU.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        resume_ckpt: Optional checkpoint path to resume from.

    Returns:
        Tuple of (trainer, val_loss, train_loss, epochs_run).
    """
    training_cfg = cfg.training

    trainer = Trainer(
        max_epochs=training_cfg.max_epochs,
        precision=training_cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        gradient_clip_val=training_cfg.gradient_clip_val,
        callbacks=build_callbacks(training_cfg),
        logger=build_loggers(training_cfg),
        num_sanity_val_steps=2,
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)

    val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
    train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
    epochs = trainer.current_epoch + 1

    return trainer, val_loss, train_loss, epochs


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the full training pipeline.

    Args:
        cfg: Hydra config injected automatically.
    """
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    gc.collect()

    training_dataset, validation_dataset, _ = create_production_datasets(
        cfg=cfg,
        dataset_path=cfg.paths.dataset_path,
    )

    num_workers: int = cfg.training.get("num_workers", 0)
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
    )

    resume_path = find_resume_checkpoint(Path("logs/tft_tb"))
    model = create_final_model(cfg, training_dataset)

    trainer, v_loss, t_loss, epochs = train_final_model(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        resume_ckpt=resume_path,
    )

    logger.info(
        "Training complete — val_loss=%.4f  train_loss=%.4f  epochs=%d",
        v_loss,
        t_loss,
        epochs,
    )

    save_production_artifacts(
        trainer=trainer,
        model=model,
        cfg=cfg,
        metrics={"val_loss": v_loss, "train_loss": t_loss, "epochs": epochs},
        dataset=training_dataset,
    )


if __name__ == "__main__":
    main()
