"""TFT model with GRU encoder / decoder and head-diversity regularisation."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_COSINE_ETA_MIN_FRACTION = 0.01


class TFTWithGRU(TemporalFusionTransformer):
    """Temporal Fusion Transformer with GRU encoder/decoder.

    Replaces default LSTM encoder and decoder with GRU cells.
    Custom kwargs.

    Args:
        head_diversity_lambda: Weight of the attention-head overlap penalty.
        reduce_on_plateau_patience: Patience for the plateau scheduler.
        scheduler: Plateau for default.
        max_epochs: Refer to config.
        **kwargs: pytorch_forecasting.TemporalFusionTransformer.
    """

    def __init__(self, **kwargs: object) -> None:

        diversity_lambda = float(kwargs.pop("head_diversity_lambda", 0.0))
        plateau_patience = int(kwargs.pop("reduce_on_plateau_patience", 2))
        scheduler_type = str(kwargs.pop("scheduler", "cosine"))
        cosine_t_max = int(kwargs.pop("max_epochs", 50))

        super().__init__(**kwargs)

        self.diversity_lambda = diversity_lambda
        self.plateau_patience = plateau_patience
        self.scheduler_type = scheduler_type
        self.cosine_t_max = cosine_t_max

        self._replace_lstm_with_gru("lstm_encoder")
        self._replace_lstm_with_gru("lstm_decoder")

    def _replace_lstm_with_gru(self, attr: str) -> None:
        """Swap LSTM module for GRU with identical hparams.

        Args:
            attr: Attribute nn.LSTM.
        """
        lstm: nn.LSTM = getattr(self, attr)
        setattr(
            self,
            attr,
            nn.GRU(
                input_size=lstm.input_size,
                hidden_size=lstm.hidden_size,
                num_layers=lstm.num_layers,
                dropout=lstm.dropout,
                batch_first=lstm.batch_first,
            ),
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW optimiser with cosine-annealing or plateau."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.get("learning_rate", 1e-3)),
            weight_decay=float(self.hparams.get("weight_decay", 0.0)),
        )

        raw_type = self.__dict__.get("scheduler_type", self.hparams.get("scheduler", "cosine"))
        scheduler_type = str(raw_type).strip().lower()
        cosine_t_max = int(self.__dict__.get("cosine_t_max", self.hparams.get("max_epochs", 50)))
        plateau_patience = int(
            self.__dict__.get(
                "plateau_patience",
                self.hparams.get("reduce_on_plateau_patience", 2)
            )
        )

        if scheduler_type != "plateau":
            lr = float(self.hparams.get("learning_rate", 1e-3))
            eta_min = lr * _COSINE_ETA_MIN_FRACTION
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max, eta_min=eta_min
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=plateau_patience,
            threshold=1e-3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def _head_overlap_loss(attn: torch.Tensor) -> torch.Tensor | None:
        """Compute mean pairwise dot-product similarity across attention heads.

        Args:
            attn: Attention weights.

        Returns:
            Scalar loss tensor or None.
        """
        if not isinstance(attn, torch.Tensor) or attn.ndim < 3:
            return None

        a: torch.Tensor = attn.mean(dim=0) if attn.ndim == 4 else attn
        num_heads = a.shape[0]
        if num_heads <= 1:
            return None

        a_flat = a.reshape(num_heads, -1)
        sim = torch.einsum("is,js->ij", a_flat, a_flat)
        mask = torch.ones(
            num_heads, num_heads, device=sim.device, dtype=torch.bool
        ).triu(diagonal=1)
        return sim[mask].sum() / (num_heads * (num_heads - 1) / 2.0)

    def step(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        batch_idx: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extend base step with optional head-diversity regularisation.

        Args:
            x: Model input dict.
            y: Target tensor.
            batch_idx: Index of current batch.
            **kwargs: Forwarded to step parent.

        Returns:
            toal_loss / output_dict as tuple.
        """
        step_out = super().step(x, y, batch_idx, **kwargs)
        base_loss, out = step_out if isinstance(step_out, tuple) else (step_out, {})

        if not self.training or self.diversity_lambda <= 0.0:
            return base_loss, out

        div_loss = self._head_overlap_loss(out.get("attention"))
        if div_loss is not None:
            self.log("head_div_loss", div_loss, on_step=False, on_epoch=True)
            return base_loss + self.diversity_lambda * div_loss, out

        return base_loss, out


def create_final_model(
    cfg: DictConfig,
    training_dataset: TimeSeriesDataSet,
) -> TFTWithGRU:
    """Instantiate TFTWithGRU from config and dataset.

    Args:
        cfg: Model / training.
        training_dataset: pytorch_forecasting.TimeSeriesDataSet.

    Returns:
        Initialised TFTWithGRU.
    """
    model = TFTWithGRU.from_dataset(
        training_dataset,
        learning_rate=float(cfg.training.learning_rate),
        hidden_size=int(cfg.model.hidden_size),
        attention_head_size=int(cfg.model.attention_head_size),
        dropout=float(cfg.model.dropout),
        hidden_continuous_size=int(cfg.model.hidden_continuous_size),
        lstm_layers=int(cfg.model.lstm_layers),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        loss=SMAPE(),
        logging_metrics=[SMAPE()],
        head_diversity_lambda=float(cfg.training.get("head_diversity_lambda", 0.0)),
        reduce_on_plateau_patience=int(cfg.training.reduce_on_plateau_patience),
        scheduler=cfg.training.get("scheduler", "cosine"),
        max_epochs=int(cfg.training.get("max_epochs", 50)),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model instantiated — %s parameters.", f"{total_params:,}")
    return model
