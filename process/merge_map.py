import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from process.utils import (
    build_weather_table,
    load_mapping,
    load_power,
    merge_power_with_weather,
    merge_weather_with_mapping,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "merge_map.yaml"
CFG = OmegaConf.load(CONFIG_PATH)

PATHS = CFG.paths
WEATHER = CFG.weather
LOGGING_CFG = CFG.logging

level_name = str(LOGGING_CFG.level).upper()
level = getattr(logging, level_name, logging.INFO)
logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_combined_power_weather_parquet(
    power_path: str,
    mapping_path: str,
    weather_glob: str,
    output_parquet: str,
    time_coord: str = "time",
) -> pd.DataFrame:
    power_df = load_power(power_path)
    mapping_df = load_mapping(mapping_path)
    weather_df = build_weather_table(weather_glob, time_coord=time_coord)
    weather_mapped_df = merge_weather_with_mapping(weather_df, mapping_df)
    combined_df = merge_power_with_weather(power_df, weather_mapped_df)
    combined_df.to_parquet(output_parquet, index=False)
    logger.info("Wrote combined power-weather data to %s", output_parquet)
    return combined_df


def main() -> None:
    power_path = str(PATHS.power_csv)
    mapping_path = str(PATHS.mapping_csv)
    weather_glob = str(PATHS.weather_glob)
    output_parquet = str(PATHS.output_parquet)
    time_coord = str(WEATHER.time_coord)

    create_combined_power_weather_parquet(
        power_path=power_path,
        mapping_path=mapping_path,
        weather_glob=weather_glob,
        output_parquet=output_parquet,
        time_coord=time_coord,
    )


if __name__ == "__main__":
    main()
