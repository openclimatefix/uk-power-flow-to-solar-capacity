"""NHiTS model wrapper."""

from __future__ import annotations

import logging

from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def create_nhits_model(
    cfg: DictConfig,
    training_dataset: TimeSeriesDataSet,
) -> NHiTS:
    """Instantiate NHiTS from config and dataset.

    Args:
        cfg: Full Hydra config with model and training sub-trees.
        training_dataset: Fitted TimeSeriesDataSet.

    Returns:
        Initialised NHiTS model.
    """
    model_cfg = cfg.model
    training_cfg = cfg.training

    model = NHiTS.from_dataset(
        training_dataset,
        learning_rate=float(training_cfg.learning_rate),
        weight_decay=float(training_cfg.weight_decay),
        hidden_size=int(model_cfg.hidden_size),
        dropout=float(model_cfg.dropout),
        n_blocks=list(model_cfg.n_blocks),
        pooling_sizes=list(model_cfg.pooling_sizes),
        downsample_frequencies=list(model_cfg.downsample_frequencies),
        backcast_loss_ratio=float(model_cfg.backcast_loss_ratio),
        log_interval=int(model_cfg.log_interval),
        loss=MAE(),
        logging_metrics=[MAE()],
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("NHiTS instantiated — %s parameters.", f"{total_params:,}")
    return model
