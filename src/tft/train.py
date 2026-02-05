import gc
import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from src.tft.data import create_production_datasets
from src.tft.model import create_final_model
from src.tft.utils import find_resume_checkpoint, save_production_artifacts

logger = logging.getLogger(__name__)


def train_final_model(
    cfg: DictConfig,
    model: pl.LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loggers: list[Logger],
    resume_ckpt: str | None = None,
) -> tuple[pl.Trainer, float, float, int]:
    training_cfg = cfg.training

    if torch.cuda.is_available() and training_cfg.get("accelerator", "gpu") == "gpu":
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=training_cfg.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
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

    trainer = pl.Trainer(
        max_epochs=training_cfg.max_epochs,
        precision=training_cfg.precision,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=training_cfg.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)

    val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
    train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
    epochs = trainer.current_epoch + 1

    return trainer, val_loss, train_loss, epochs


@hydra.main(version_base=None, config_path="../../configs/tft", config_name="tft_model")
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    gc.collect()

    training_dataset, validation_dataset, _ = create_production_datasets(
        cfg=cfg,
        dataset_path=cfg.paths.dataset_path,
    )

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 0),
    )

    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 0),
    )

    tb_logger = TensorBoardLogger(save_dir="logs", name="tft_tb")
    csv_logger = CSVLogger(save_dir="logs", name="tft_csv")

    resume_path = find_resume_checkpoint(Path("logs/tft_tb"))

    model = create_final_model(cfg, training_dataset)

    trainer, v_loss, t_loss, epochs = train_final_model(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loggers=[tb_logger, csv_logger],
        resume_ckpt=resume_path,
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
