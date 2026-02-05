import logging
from pathlib import Path

import pandas as pd

from process.preprocess import (
    COMBINED_FULL_CSV,
    COMBINED_POWER_ONLY_CSV,
    COMBINED_REDUCED_CSV,
    COMBINED_FILLED_POWER_CSV,
    COMBINED_AGG_LOCATION_CSV,
)
from process.utils import analyze_directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def check_file_exists(path: str | Path) -> bool:
    p = Path(path)
    exists = p.exists()
    if exists:
        logger.info("File exists: %s (size=%.2f MB)", p, p.stat().st_size / (1024 * 1024))
    else:
        logger.warning("Missing expected file: %s", p)
    return exists


def check_csv_basic(path: str | Path, nrows: int = 100) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        logger.warning("Cannot inspect CSV, file missing: %s", p)
        return None
    try:
        df = pd.read_csv(p, nrows=nrows)
        logger.info(
            "Loaded sample from %s: %d rows, %d columns. Columns: %s",
            p,
            len(df),
            df.shape[1],
            list(df.columns),
        )
        return df
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", p, exc)
        return None


def run_sanity_checks() -> None:
    logger.info("Running sanity checks on preprocessed artifacts")

    check_file_exists(COMBINED_FULL_CSV)
    check_file_exists(COMBINED_POWER_ONLY_CSV)
    check_file_exists(COMBINED_REDUCED_CSV)
    check_file_exists(COMBINED_FILLED_POWER_CSV)
    check_file_exists(COMBINED_AGG_LOCATION_CSV)

    df_full_sample = check_csv_basic(COMBINED_FULL_CSV)
    if df_full_sample is not None:
        if "tx_id" in df_full_sample.columns and "hh" in df_full_sample.columns:
            logger.info(
                "FULL sample has tx_id and hh; unique tx_ids (sample): %d",
                df_full_sample["tx_id"].nunique(),
            )
        else:
            logger.warning("FULL sample missing expected columns 'tx_id' and/or 'hh'")

    df_power_only_sample = check_csv_basic(COMBINED_POWER_ONLY_CSV)
    if df_power_only_sample is not None:
        missing = [c for c in ("tx_id", "hh", "kva") if c not in df_power_only_sample.columns]
        if missing:
            logger.warning(
                "POWER_ONLY sample missing expected columns: %s",
                missing,
            )
        else:
            logger.info(
                "POWER_ONLY sample looks OK, non-null kva fraction: %.3f",
                df_power_only_sample["kva"].notna().mean(),
            )

    df_filled_sample = check_csv_basic(COMBINED_FILLED_POWER_CSV)
    if df_filled_sample is not None:
        cols_ok = all(c in df_filled_sample.columns for c in ("tx_id", "hh", "active_power_kW"))
        if not cols_ok:
            logger.warning("FILLED sample missing one of: tx_id, hh, active_power_kW")
        else:
            non_null_frac = df_filled_sample["active_power_kW"].notna().mean()
            logger.info(
                "FILLED sample active_power_kW non-null fraction: %.3f",
                non_null_frac,
            )

    df_agg_sample = check_csv_basic(COMBINED_AGG_LOCATION_CSV)
    if df_agg_sample is not None:
        expected_cols = ("tx_id", "hh", "active_power_kW")
        missing = [c for c in expected_cols if c not in df_agg_sample.columns]
        if missing:
            logger.warning(
                "AGG_LOCATION sample missing expected columns: %s",
                missing,
            )
        else:
            logger.info(
                "AGG_LOCATION sample ok; distinct tx_id (sample): %d",
                df_agg_sample["tx_id"].nunique(),
            )

    logger.info("Sanity checks completed")


def main() -> None:
    analyze_directory(Path(".").resolve().as_posix())
    run_sanity_checks()


if __name__ == "__main__":
    main()
