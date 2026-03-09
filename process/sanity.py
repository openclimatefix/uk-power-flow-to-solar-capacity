"""Sanity checks for preprocessed UKPN power demand artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.process.utils import analyze_directory

logger = logging.getLogger(__name__)


def check_file_exists(path: str | Path) -> bool:
    """Log existence and size of a file.

    Args:
        path: File path to check.

    Returns:
        True if the file exists, False otherwise.
    """
    p = Path(path)
    if p.exists():
        logger.info("File exists: %s (size=%.2f MB)", p, p.stat().st_size / (1024 * 1024))
        return True
    logger.warning("Missing expected file: %s", p)
    return False


def check_csv_basic(path: str | Path, nrows: int = 100) -> pd.DataFrame | None:
    """Load a sample from a CSV and log its shape and columns.

    Args:
        path: Path to the CSV file.
        nrows: Number of rows to sample.

    Returns:
        Sample DataFrame, or None if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Cannot inspect CSV, file missing: %s", p)
        return None
    df = pd.read_csv(p, nrows=nrows)
    logger.info(
        "Loaded sample from %s: %d rows, %d columns. Columns: %s",
        p, len(df), df.shape[1], list(df.columns),
    )
    return df


def run_sanity_checks(cfg: DictConfig) -> None:
    """Run existence and schema checks on all preprocessed output files.

    Args:
        cfg: Full Hydra config.
    """
    paths = cfg.paths
    logger.info("Running sanity checks on preprocessed artifacts")

    for key in (
        "combined_full_csv",
        "combined_power_only_csv",
        "combined_reduced_csv",
        "combined_filled_power_csv",
        "combined_aggregated_location_csv",
    ):
        check_file_exists(str(getattr(paths, key)))

    df_full_sample = check_csv_basic(paths.combined_full_csv)
    if df_full_sample is not None:
        if "tx_id" in df_full_sample.columns and "hh" in df_full_sample.columns:
            logger.info(
                "FULL sample has tx_id and hh; unique tx_ids (sample): %d",
                df_full_sample["tx_id"].nunique(),
            )
        else:
            logger.warning("FULL sample missing expected columns 'tx_id' and/or 'hh'")

    df_power_only_sample = check_csv_basic(paths.combined_power_only_csv)
    if df_power_only_sample is not None:
        missing = [c for c in ("tx_id", "hh", "kva") if c not in df_power_only_sample.columns]
        if missing:
            logger.warning("POWER_ONLY sample missing expected columns: %s", missing)
        else:
            logger.info(
                "POWER_ONLY sample looks OK, non-null kva fraction: %.3f",
                df_power_only_sample["kva"].notna().mean(),
            )

    df_filled_sample = check_csv_basic(paths.combined_filled_power_csv)
    if df_filled_sample is not None:
        if not all(c in df_filled_sample.columns for c in ("tx_id", "hh", "active_power_kW")):
            logger.warning("FILLED sample missing one of: tx_id, hh, active_power_kW")
        else:
            logger.info(
                "FILLED sample active_power_kW non-null fraction: %.3f",
                df_filled_sample["active_power_kW"].notna().mean(),
            )

    df_agg_sample = check_csv_basic(paths.combined_aggregated_location_csv)
    if df_agg_sample is not None:
        missing = [
            c for c in ("tx_id", "hh", "active_power_kW") if c not in df_agg_sample.columns
        ]
        if missing:
            logger.warning("AGG_LOCATION sample missing expected columns: %s", missing)
        else:
            logger.info(
                "AGG_LOCATION sample ok; distinct tx_id (sample): %d",
                df_agg_sample["tx_id"].nunique(),
            )

    logger.info("Sanity checks completed")


@hydra.main(version_base=None, config_path="../../configs/process", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the sanity check pipeline.

    Args:
        cfg: Hydra config injected automatically.
    """
    analyze_directory(str(Path(cfg.paths.base_data_dir).resolve()))
    run_sanity_checks(cfg)


if __name__ == "__main__":
    main()
