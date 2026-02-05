import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.tft.model import TFTWithGRU, create_final_model


@pytest.fixture
def mock_cfg() -> DictConfig:
    """Provides a mock Hydra configuration for model instantiation."""
    return OmegaConf.create({
        "model": {
            "hidden_size": 16,
            "attention_head_size": 4,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "lstm_layers": 2,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "reduce_on_plateau_patience": 2,
            "head_diversity_lambda": 0.1,
        }
    })


def _make_dummy_dataset() -> TimeSeriesDataSet:
    """Creates a minimal TimeSeriesDataSet for testing."""
    data = {
        "time_idx": list(range(5)) + list(range(5)),
        "location": ["site_1"] * 5 + ["site_2"] * 5,
        "active_power_mw_clean": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
        "u10": [1.0] * 10,
        "v10": [0.0] * 10,
    }
    df = pd.DataFrame(data)

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="active_power_mw_clean",
        group_ids=["location"],
        max_encoder_length=2,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=["u10", "v10"],
        time_varying_unknown_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(groups=["location"], transformation="softplus"),
    )
    return dataset


def test_create_final_model_builds_tft_with_gru(mock_cfg: DictConfig) -> None:
    """Verifies that the model is instantiated as TFTWithGRU and swaps LSTMs correctly."""
    training_dataset = _make_dummy_dataset()
    model = create_final_model(mock_cfg, training_dataset)

    assert isinstance(model, TFTWithGRU)
    # Verify swapped to GRU
    assert isinstance(model.lstm_encoder, torch.nn.GRU)
    assert isinstance(model.lstm_decoder, torch.nn.GRU)

    # Verify hparams match config
    assert model.hparams.learning_rate == mock_cfg.training.learning_rate
    assert model.hparams.hidden_size == mock_cfg.model.hidden_size


def test_configure_optimizers_uses_adamw_and_scheduler(mock_cfg: DictConfig) -> None:
    """Checks that AdamW and the ReduceLROnPlateau scheduler are configured."""
    training_dataset = _make_dummy_dataset()
    model = create_final_model(mock_cfg, training_dataset)

    opt_conf = model.configure_optimizers()
    optimizer = opt_conf["optimizer"]
    scheduler_dict = opt_conf["lr_scheduler"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == mock_cfg.training.learning_rate
    assert "scheduler" in scheduler_dict
    assert isinstance(scheduler_dict["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_head_overlap_loss_calculation() -> None:
    """Tests the static attention diversity loss calculation."""
    # Create dummy attention tensor (Batch=2, Heads=2, Time=2, Time=2)
    # Two identical heads should produce higher overlap loss
    attn = torch.ones((2, 2, 2, 2))
    loss = TFTWithGRU._head_overlap_loss(attn)

    assert loss is not None
    assert loss > 0

    # Single head should return None
    single_head_attn = torch.ones((2, 1, 2, 2))
    assert TFTWithGRU._head_overlap_loss(single_head_attn) is None
