import logging
from typing import Any

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TFTWithGRU(TemporalFusionTransformer):
    def __init__(
        self,
        **kwargs: Any
    ) -> None:
        self.head_diversity_lambda = kwargs.pop("head_diversity_lambda", 0.0)
        self.reduce_on_plateau_patience = kwargs.pop("reduce_on_plateau_patience", 2)

        super().__init__(**kwargs)

        self.hparams.update(
            {
                "head_diversity_lambda": self.head_diversity_lambda,
                "reduce_on_plateau_patience": self.reduce_on_plateau_patience,
            }
        )

        input_size = self.lstm_encoder.input_size
        hidden_size = self.lstm_encoder.hidden_size
        num_layers = self.lstm_encoder.num_layers
        dropout = self.lstm_encoder.dropout
        batch_first = self.lstm_encoder.batch_first

        self.lstm_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.lstm_decoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )


    def configure_optimizers(self) -> dict[str, Any]:
        weight_decay = getattr(self.hparams, "weight_decay", 0.0)
        learning_rate = getattr(self.hparams, "learning_rate", 1e-3)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(self.hparams.get("reduce_on_plateau_patience", 2)),
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
        if not isinstance(attn, torch.Tensor) or attn.ndim < 3:
            return None

        attn_h = attn.mean(dim=0) if attn.ndim == 4 else attn
        num_heads = attn_h.shape[0]

        if num_heads <= 1:
            return None

        div = attn_h.new_zeros(())
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                div = div + torch.mean(attn_h[i] * attn_h[j])

        return div / (num_heads * (num_heads - 1) / 2.0)

    def step(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        batch_idx: int,
        **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        step_out = super().step(x, y, batch_idx, **kwargs)
        base_loss, out = step_out if isinstance(step_out, tuple) else (step_out, {})

        if not self.training:
            return base_loss, out

        lam = float(self.hparams.get("head_diversity_lambda", 0.0))
        if lam <= 0:
            return base_loss, out

        attn = out.get("attention")
        div_loss = self._head_overlap_loss(attn)

        if div_loss is not None:
            total_loss = base_loss + (lam * div_loss)
            self.log("head_div_loss", div_loss, on_step=False, on_epoch=True)
            return total_loss, out

        return base_loss, out


def create_final_model(cfg: DictConfig, training_dataset: TimeSeriesDataSet) -> TFTWithGRU:
    m: TFTWithGRU = TFTWithGRU.from_dataset(
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
        reduce_on_plateau_patience=cfg.training.reduce_on_plateau_patience,
        head_diversity_lambda=cfg.training.get("head_diversity_lambda", 0.0),
    )

    total_params = sum(p.numel() for p in m.parameters())
    logger.info(f"Model instantiated with {total_params:,} parameters.")

    return m
