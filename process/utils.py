import glob
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_preprocess_config(config_path: str | None = None) -> DictConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "preprocess.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info("Loaded preprocess config from %s", config_path)
    return cfg


def load_merge_map_config(config_path: str | None = None) -> DictConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "merge_map.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info("Loaded merge_map config from %s", config_path)
    return cfg


def get_path(cfg: DictConfig, key: str) -> str:
    return str(cfg.paths[key])


def get_processing_param(cfg: DictConfig, key: str):
    return cfg.processing[key]


def get_matching_param(cfg: DictConfig, key: str):
    return cfg.matching[key]


def get_plot_param(cfg: DictConfig, key: str, plot_name: str = "ten_locations"):
    return cfg.plots[plot_name][key]


def get_mm_path(cfg: DictConfig, key: str) -> str:
    return str(cfg.paths[key])


def get_weather_param(cfg: DictConfig, key: str):
    return cfg.weather[key]


def get_mapping_param(cfg: DictConfig, key: str):
    return cfg.mapping[key]


def get_power_param(cfg: DictConfig, key: str):
    return cfg.power[key]


def get_merge_param(cfg: DictConfig, key: str):
    return cfg.merge[key]


def get_merge_logging_level(cfg: DictConfig) -> str:
    return str(cfg.logging.level)


def analyze_directory(directory: str) -> None:
    file_count = 0
    folder_count = 0
    file_types = defaultdict(int)
    file_sizes = []
    total_size = 0

    for root, dirs, files in os.walk(directory):
        folder_count += len(dirs)
        for file in files:
            file_count += 1
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            if ext:
                file_types[ext.lower()] += 1
            else:
                file_types["no_extension"] += 1
            size = os.path.getsize(file_path)
            file_sizes.append(size)
            total_size += size

    logger.info(
        "Directory %s: %d folders, %d files, %.2f GB",
        directory,
        folder_count,
        file_count,
        total_size / (1024**3),
    )
    if file_sizes:
        logger.info("Average file size %.2f MB", np.mean(file_sizes) / (1024**2))
        logger.info("Largest file size %.2f MB", np.max(file_sizes) / (1024**2))
        logger.info("Smallest file size %.2f MB", np.min(file_sizes) / (1024**2))


def extract_zip(zip_path: str, extract_to: str) -> None:
    if not os.path.exists(extract_to):
        import zipfile

        logger.info("Extracting %s to %s", zip_path, extract_to)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        logger.info("Extraction completed")
    else:
        logger.info("Extract path %s already exists, skipping extraction", extract_to)


def analyze_top_level_folders(extract_path: str) -> None:
    top_folders = defaultdict(int)
    for root, _dirs, files in os.walk(extract_path):
        rel_root = os.path.relpath(root, extract_path)
        top_level = rel_root.split(os.sep)[0]
        top_folders[top_level] += len(files)
    for folder, file_count in sorted(top_folders.items(), key=lambda x: x[1], reverse=True):
        logger.info("Top-level folder %s: %d files", folder, file_count)


def find_data_files(
    extract_path: str,
    data_extensions: Iterable[str] | None = None,
) -> list[str]:
    if data_extensions is None:
        data_extensions = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".h5", ".nc"}
    ext_set = {e.lower() for e in data_extensions}
    data_files: list[str] = []
    for root, _dirs, files in os.walk(extract_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in ext_set:
                rel_path = os.path.relpath(os.path.join(root, file), extract_path)
                data_files.append(rel_path)
    logger.info("Found %d data files under %s", len(data_files), extract_path)
    return data_files


def basic_string_cleanup(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def jaccard_token_overlap(str1: str, str2: str) -> float:
    s1 = set(basic_string_cleanup(str1).split())
    s2 = set(basic_string_cleanup(str2).split())
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


def load_mapping(mapping_path: str) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path)
    if not {"era5_lat", "era5_lon"}.issubset(mapping_df.columns):
        raise ValueError("Mapping file must contain 'era5_lat' and 'era5_lon' columns")
    logger.info("Loaded mapping from %s with %d rows", mapping_path, len(mapping_df))
    return mapping_df


def load_power(power_path: str) -> pd.DataFrame:
    power_df = pd.read_csv(power_path, parse_dates=["timestamp"])
    if "location" not in power_df.columns:
        raise ValueError("Power data must contain 'location' column")
    logger.info("Loaded power data from %s with %d rows", power_path, len(power_df))
    return power_df


def load_weather_files(weather_glob: str) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(weather_glob))
    if not files:
        raise FileNotFoundError(f"No weather files found for pattern: {weather_glob}")
    logger.info("Found %d weather files matching %s", len(files), weather_glob)
    return files


def weather_ds_to_dataframe(ds: xr.Dataset, time_coord: str = "time") -> pd.DataFrame:
    if time_coord not in ds.coords:
        raise ValueError(f"Expected time coordinate '{time_coord}' in dataset")

    data_vars = list(ds.data_vars)
    df = ds[data_vars].to_dataframe().reset_index()

    rename_dict = {time_coord: "timestamp"}
    rename_dict.update({var: f"weather_{var}" for var in data_vars})
    df = df.rename(columns=rename_dict)

    if "latitude" in df.columns:
        df = df.rename(columns={"latitude": "era5_lat"})
    elif "lat" in df.columns:
        df = df.rename(columns={"lat": "era5_lat"})

    if "longitude" in df.columns:
        df = df.rename(columns={"longitude": "era5_lon"})
    elif "lon" in df.columns:
        df = df.rename(columns={"lon": "era5_lon"})

    logger.info("Converted dataset with %d rows to DataFrame", len(df))
    return df


def build_weather_table(weather_glob: str, time_coord: str = "time") -> pd.DataFrame:
    files = load_weather_files(weather_glob)
    frames: list[pd.DataFrame] = []
    for path in files:
        logger.info("Loading weather file %s", path)
        with xr.open_dataset(path) as ds:
            df = weather_ds_to_dataframe(ds, time_coord=time_coord)
            frames.append(df)
    weather_df = pd.concat(frames, ignore_index=True)
    logger.info("Built weather table with %d rows from %d files", len(weather_df), len(files))
    return weather_df


def merge_weather_with_mapping(weather_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["era5_lat", "era5_lon"]
    for col in join_cols:
        if col not in weather_df.columns or col not in mapping_df.columns:
            raise ValueError(f"Expected column '{col}' in both weather and mapping data")

    merged = weather_df.merge(mapping_df, on=join_cols, how="inner")
    if "location" not in merged.columns:
        logger.warning("Result of merging weather and mapping has no 'location' column")
    logger.info("Merged weather with mapping, resulting rows: %d", len(merged))
    return merged


def merge_power_with_weather(power_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in power_df.columns:
        raise ValueError("Power data must have 'timestamp' column")
    if "timestamp" not in weather_df.columns:
        raise ValueError("Weather data must have 'timestamp' column")
    if "location" not in power_df.columns or "location" not in weather_df.columns:
        raise ValueError("Both power and weather data must have 'location' column")

    merged = power_df.merge(weather_df, on=["location", "timestamp"], how="inner")
    logger.info(
        "Merged power with weather on location and timestamp, resulting rows: %d",
        len(merged),
    )
    return merged
