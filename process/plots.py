import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _load_default_filepath_from_config() -> str | None:
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "configs" / "preprocess.yaml"
    cfg = OmegaConf.load(cfg_path)
    return cfg.paths.combined_aggregated_location_csv


def plot_ten_locations_fixed(
    filepath: str | None = None,
    window_days: int = 30,
    downsample_step: int = 3,
    max_locations: int = 10,
) -> None:
    if filepath is None:
        filepath = _load_default_filepath_from_config()
        if filepath is None:
            raise ValueError(
                "filepath is None and could not be loaded from configs/preprocess.yaml "
                "(expected key: paths.combined_aggregated_location_csv)"
            )

    df = pd.read_csv(filepath, parse_dates=["hh"])
    if "tx_id" not in df.columns:
        raise ValueError("Expected tx_id column in aggregated dataset")

    logger.info("Loaded aggregated data from %s with %d rows", filepath, len(df))

    locations = df["tx_id"].unique()
    if len(locations) > max_locations:
        selected_locations = np.random.choice(locations, size=max_locations, replace=False)
    else:
        selected_locations = locations

    df = df.sort_values(by="hh")
    start_date = df["hh"].min()
    window_end = start_date + pd.Timedelta(days=window_days)
    df_window = df[(df["hh"] >= start_date) & (df["hh"] < window_end)]

    logger.info(
        "Plotting %d locations over %d-day window from %s to %s",
        len(selected_locations),
        window_days,
        start_date,
        window_end,
    )

    n_plots = len(selected_locations)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = [axes] if n_plots == 1 else axes.flatten()

    for i, loc in enumerate(selected_locations):
        ax = axes[i]
        loc_data = df_window[df_window["tx_id"] == loc].copy()
        if loc_data.empty:
            ax.set_title(str(loc))
            continue
        if downsample_step > 1:
            loc_data = loc_data.iloc[::downsample_step, :]
        ax.plot(loc_data["hh"], loc_data["active_power_kW"], linewidth=0.8)
        ax.set_title(str(loc))
        ax.set_ylabel("Active Power (kW)")
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.autofmt_xdate()
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

    logger.info("Finished plotting locations from %s", filepath)
