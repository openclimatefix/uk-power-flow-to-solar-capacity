from __future__ import annotations

from unittest.mock import patch

import torch
from pytorch_forecasting import TimeSeriesDataSet

import pytest
from omegaconf import DictConfig

from src.tft.model import TFTWithGRU, create_final_model


def test_cosine_scheduler(model: TFTWithGRU) -> None:
    model.__dict__["scheduler_type"] = "cosine"
    model.__dict__["cosine_t_max"] = 10
    sched = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_plateau_scheduler(model: TFTWithGRU) -> None:
    model.__dict__["scheduler_type"] = "plateau"
    model.__dict__["plateau_patience"] = 2
    sched = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    model.__dict__["scheduler_type"] = "cosine"


def test_overlap_loss_positive_for_identical_heads() -> None:
    loss = TFTWithGRU._head_overlap_loss(torch.ones(2, 2, 4, 4))
    assert loss is not None
    assert loss.item() > 0


def test_overlap_loss_zero_for_orthogonal_heads() -> None:
    attn = torch.zeros(2, 4, 4)
    attn[0, 0, 0] = 1.0
    attn[1, 1, 1] = 1.0
    loss = TFTWithGRU._head_overlap_loss(attn)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_overlap_loss_none_for_single_head() -> None:
    assert TFTWithGRU._head_overlap_loss(torch.ones(1, 4, 4)) is None


def test_overlap_loss_none_for_non_tensor() -> None:
    assert TFTWithGRU._head_overlap_loss(None) is None


def test_step_adds_penalty_during_training(
    cfg: DictConfig, dataset: TimeSeriesDataSet
) -> None:
    m = create_final_model(cfg, dataset)
    m.__dict__["diversity_lambda"] = 1.0
    m.train()
    base_loss = torch.tensor(1.0)
    parent_out = (base_loss, {"attention": torch.ones(2, 2, 4, 4)})

    with (
        patch.object(type(m).__mro__[1], "step", return_value=parent_out),
        patch.object(m, "log"),
    ):
        result, _ = m.step({}, torch.zeros(2), 0)

    assert result.item() > 1.0


def test_step_no_penalty_during_eval(
    cfg: DictConfig, dataset: TimeSeriesDataSet
) -> None:
    m = create_final_model(cfg, dataset)
    m.eval()
    base_loss = torch.tensor(2.5)
    parent_out = (base_loss, {"attention": torch.ones(2, 2, 4, 4)})

    with patch.object(type(m).__mro__[1], "step", return_value=parent_out):
        result, _ = m.step({}, torch.zeros(2), 0)

    assert result.item() == pytest.approx(2.5)
