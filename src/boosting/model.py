"""XGBoost model wrapper."""

from __future__ import annotations

import logging

import xgboost as xgb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def create_xgboost_model(cfg: DictConfig) -> xgb.XGBRegressor:
    """Instantiate XGBRegressor from config.

    Args:
        cfg: Full Hydra config with model and training sub-trees.

    Returns:
        Initialised XGBRegressor.
    """
    model_cfg = cfg.model
    training_cfg = cfg.training

    model = xgb.XGBRegressor(
        n_estimators=int(model_cfg.n_estimators),
        max_depth=int(model_cfg.max_depth),
        learning_rate=float(model_cfg.learning_rate),
        subsample=float(model_cfg.subsample),
        colsample_bytree=float(model_cfg.colsample_bytree),
        min_child_weight=int(model_cfg.min_child_weight),
        gamma=float(model_cfg.gamma),
        reg_alpha=float(model_cfg.reg_alpha),
        reg_lambda=float(model_cfg.reg_lambda),
        tree_method=str(model_cfg.tree_method),
        device=str(model_cfg.device),
        eval_metric=str(training_cfg.eval_metric),
        early_stopping_rounds=int(training_cfg.early_stopping_rounds),
        n_jobs=int(training_cfg.get("num_workers", 4)),
    )

    logger.info(
        "XGBRegressor instantiated — n_estimators=%d, max_depth=%d, lr=%.4f",
        model_cfg.n_estimators,
        model_cfg.max_depth,
        model_cfg.learning_rate,
    )
    return model
