"""Pipeline for merging power demand data with ERA5 weather and site mapping."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from src.process.utils import (
    build_weather_table,
    load_mapping,
    load_power,
    merge_power_with_weather,
    merge_weather_with_mapping,
)

logger = logging.getLogger(__name__)


def create_combined_power_weather_parquet(
    power_path: str,
    mapping_path: str,
    weather_glob: str,
    output_parquet: str,
    time_coord: str = "time",
) -> pd.DataFrame:
    """Merge power, mapping, and weather data into a single Parquet file.

    Args:
        power_path: Path to the power CSV file.
        mapping_path: Path to the site-to-ERA5 mapping CSV.
        weather_glob: Glob pattern for ERA5 NetCDF files.
        output_parquet: Destination Parquet file path.
        time_coord: Name of the time coordinate in the NetCDF files.

    Returns:
        Combined DataFrame written to output_parquet.
    """
    power_df = load_power(power_path)
    mapping_df = load_mapping(mapping_path)
    weather_df = build_weather_table(weather_glob, time_coord=time_coord)
    weather_mapped_df = merge_weather_with_mapping(weather_df, mapping_df)
    combined_df = merge_power_with_weather(power_df, weather_mapped_df)

    table = pa.Table.from_pandas(combined_df, preserve_index=False)
    pq.write_table(table, output_parquet, compression="snappy")
    logger.info("Wrote combined power-weather data to %s", output_parquet)

    return combined_df


@hydra.main(version_base=None, config_path="../../configs/process", config_name="merge_map")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the power-weather merge pipeline.

    Args:
        cfg: Hydra config injected automatically.

    Raises:
        FileNotFoundError: If power CSV or mapping CSV does not exist.
    """
    power_path = Path(cfg.paths.power_csv)
    mapping_path = Path(cfg.paths.mapping_csv)

    if not power_path.exists():
        raise FileNotFoundError(f"Power CSV not found: {power_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")

    logger.info("Starting power-weather merge pipeline.")

    create_combined_power_weather_parquet(
        power_path=str(power_path),
        mapping_path=str(mapping_path),
        weather_glob=str(cfg.paths.weather_glob),
        output_parquet=str(cfg.paths.output_parquet),
        time_coord=str(cfg.weather.time_coord),
    )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
