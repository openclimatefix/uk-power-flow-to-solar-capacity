"""Full training pipeline for XGBoost model."""

from __future__ import annotations

import logging

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from src.xgboost.data import create_train_val_split
from src.xgboost.model import create_xgboost_model
from src.xgboost.utils import (
    compute_metrics,
    log_feature_importance,
    save_xgboost_artifacts,
)

logger = logging.getLogger(__name__)


def train_xgboost_model(
    cfg: DictConfig,
) -> tuple[object, dict[str, float], list[str]]:
    """Load data, fit XGBoost, evaluate, and return results.

    Args:
        cfg: Full Hydra config.

    Returns:
        Tuple of (model, metrics, feature_cols).
    """
    train_df, val_df, feature_cols = create_train_val_split(
        cfg=cfg,
        dataset_path=cfg.paths.dataset_path,
    )

    target = cfg.model.target

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target].to_numpy(dtype=np.float32)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target].to_numpy(dtype=np.float32)

    model = create_xgboost_model(cfg)

    logger.info("Fitting XGBoost model...")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=100,
    )

    y_pred = model.predict(x_val)
    metrics = compute_metrics(y_val, y_pred)

    logger.info(
        "Validation — MAE=%.4f  RMSE=%.4f  SMAPE=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["smape"],
    )

    log_feature_importance(model, feature_cols)

    return model, metrics, feature_cols


@hydra.main(version_base=None, config_path="../../configs/xgboost", config_name="boosting")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the XGBoost training pipeline.

    Args:
        cfg: Hydra config injected automatically.
    """
    tracking_uri = cfg.training.get("mlflow_tracking_uri")

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(str(cfg.training.get("mlflow_experiment", "xgboost")))

        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": cfg.model.n_estimators,
                "max_depth": cfg.model.max_depth,
                "learning_rate": cfg.model.learning_rate,
                "subsample": cfg.model.subsample,
                "colsample_bytree": cfg.model.colsample_bytree,
            })

            model, metrics, feature_cols = train_xgboost_model(cfg)

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="model")

    else:
        model, metrics, feature_cols = train_xgboost_model(cfg)

    save_xgboost_artifacts(
        model=model,
        cfg=cfg,
        metrics=metrics,
        feature_cols=feature_cols,
        output_dir=str(cfg.paths.results_path),
    )

    logger.info(
        "Training complete — MAE=%.4f  RMSE=%.4f  SMAPE=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["smape"],
    )


if __name__ == "__main__":
    main()
