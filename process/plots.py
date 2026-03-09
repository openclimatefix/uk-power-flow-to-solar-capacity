"""Plotting utilities for preprocessed power demand data."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def plot_ten_locations_fixed(
    cfg: DictConfig,
    filepath: str | None = None,
) -> None:
    """Plot active power time series for a sample of locations.

    Args:
        cfg: Full Hydra config.
        filepath: Path to the aggregated location CSV. Defaults to
            cfg.paths.combined_aggregated_location_csv.

    Raises:
        ValueError: If the loaded DataFrame is missing the tx_id column.
    """
    plot_cfg = cfg.plots.ten_locations

    if filepath is None:
        filepath = str(cfg.paths.combined_aggregated_location_csv)

    df = pd.read_csv(filepath, parse_dates=["hh"])
    if "tx_id" not in df.columns:
        raise ValueError("Expected tx_id column in aggregated dataset")

    logger.info("Loaded aggregated data from %s with %d rows", filepath, len(df))

    locations = df["tx_id"].unique()
    max_locations = int(plot_cfg.max_locations)
    selected_locations = (
        np.random.choice(locations, size=max_locations, replace=False)
        if len(locations) > max_locations
        else locations
    )

    df = df.sort_values(by="hh")
    start_date = df["hh"].min()
    window_end = start_date + pd.Timedelta(days=int(plot_cfg.window_days))
    df_window = df[(df["hh"] >= start_date) & (df["hh"] < window_end)]

    logger.info(
        "Plotting %d locations over %d-day window from %s to %s",
        len(selected_locations),
        plot_cfg.window_days,
        start_date,
        window_end,
    )

    n_plots = len(selected_locations)
    n_cols = int(plot_cfg.n_cols)
    n_rows = int(np.ceil(n_plots / n_cols))
    figsize = (
        float(plot_cfg.figsize_base_width),
        float(plot_cfg.figsize_height_per_row) * n_rows,
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = [axes] if n_plots == 1 else axes.flatten()

    downsample_step = int(plot_cfg.downsample_step)

    for i, loc in enumerate(selected_locations):
        ax = axes[i]
        loc_data = df_window[df_window["tx_id"] == loc].copy()
        if loc_data.empty:
            ax.set_title(str(loc))
            continue
        if downsample_step > 1:
            loc_data = loc_data.iloc[::downsample_step, :]
        ax.plot(
            loc_data["hh"],
            loc_data["active_power_kW"],
            linewidth=float(plot_cfg.line_width),
        )
        ax.set_title(str(loc))
        ax.set_ylabel("Active Power (kW)")
        ax.grid(
            True,
            linestyle=str(plot_cfg.grid_linestyle),
            alpha=float(plot_cfg.grid_alpha),
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.autofmt_xdate()
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

    logger.info("Finished plotting locations from %s", filepath)
