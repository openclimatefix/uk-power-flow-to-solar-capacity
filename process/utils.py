"""Utility functions for data loading, preprocessing, and merging."""

from __future__ import annotations

import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def analyze_directory(directory: str) -> None:
    """Log file count, type breakdown, and size statistics for a directory tree.

    Args:
        directory: Root directory to walk.
    """
    file_count = 0
    folder_count = 0
    file_types: dict[str, int] = defaultdict(int)
    file_sizes: list[int] = []
    total_size = 0

    for path in Path(directory).rglob("*"):
        if path.is_dir():
            folder_count += 1
        elif path.is_file():
            file_count += 1
            ext = path.suffix.lower() or "no_extension"
            file_types[ext] += 1
            size = path.stat().st_size
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
    """Extract a zip archive if the destination does not already exist.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract into.
    """
    dest = Path(extract_to)
    if dest.exists():
        logger.info("Extract path %s already exists, skipping extraction", extract_to)
        return
    logger.info("Extracting %s to %s", zip_path, extract_to)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info("Extraction completed")


def analyze_top_level_folders(extract_path: str) -> None:
    """Log per-top-level-folder file counts under an extracted directory.

    Args:
        extract_path: Root of the extracted archive.
    """
    top_folders: dict[str, int] = defaultdict(int)
    root = Path(extract_path)
    for path in root.rglob("*"):
        if path.is_file():
            top_level = path.relative_to(root).parts[0]
            top_folders[top_level] += 1
    for folder, count in sorted(top_folders.items(), key=lambda x: x[1], reverse=True):
        logger.info("Top-level folder %s: %d files", folder, count)


def find_data_files(
    extract_path: str,
    data_extensions: Iterable[str] | None = None,
) -> list[str]:
    """Return relative paths of data files found under extract_path.

    Args:
        extract_path: Root directory to search.
        data_extensions: File extensions to match. Defaults to common data formats.

    Returns:
        Sorted list of relative file path strings.
    """
    if data_extensions is None:
        data_extensions = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".h5", ".nc"}
    ext_set = {e.lower() for e in data_extensions}
    root = Path(extract_path)
    data_files = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in ext_set
    ]
    logger.info("Found %d data files under %s", len(data_files), extract_path)
    return sorted(data_files)


def basic_string_cleanup(s: str) -> str:
    """Lowercase, strip, and normalise whitespace in a string.

    Args:
        s: Input string.

    Returns:
        Cleaned string.
    """
    return " ".join(str(s).strip().lower().split())


def jaccard_token_overlap(str1: str, str2: str) -> float:
    """Compute Jaccard similarity between the token sets of two strings.

    Args:
        str1: First string.
        str2: Second string.

    Returns:
        Float in [0, 1]; 0.0 if either string is empty after cleanup.
    """
    s1 = set(basic_string_cleanup(str1).split())
    s2 = set(basic_string_cleanup(str2).split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def load_mapping(mapping_path: str) -> pd.DataFrame:
    """Load the site-to-ERA5 mapping CSV.

    Args:
        mapping_path: Path to the mapping CSV.

    Returns:
        DataFrame with at least era5_lat and era5_lon columns.

    Raises:
        ValueError: If required columns are absent.
    """
    mapping_df = pd.read_csv(mapping_path)
    if not {"era5_lat", "era5_lon"}.issubset(mapping_df.columns):
        raise ValueError("Mapping file must contain 'era5_lat' and 'era5_lon' columns")
    logger.info("Loaded mapping from %s with %d rows", mapping_path, len(mapping_df))
    return mapping_df


def load_power(power_path: str) -> pd.DataFrame:
    """Load the power demand CSV.

    Args:
        power_path: Path to the power CSV.

    Returns:
        DataFrame with parsed timestamp column.

    Raises:
        ValueError: If location column is absent.
    """
    power_df = pd.read_csv(power_path, parse_dates=["timestamp"])
    if "location" not in power_df.columns:
        raise ValueError("Power data must contain 'location' column")
    logger.info("Loaded power data from %s with %d rows", power_path, len(power_df))
    return power_df


def load_weather_files(weather_glob: str) -> list[Path]:
    """Return sorted list of NetCDF paths matching a glob pattern.

    Args:
        weather_glob: Glob pattern for ERA5 NetCDF files.

    Returns:
        Sorted list of Path objects.

    Raises:
        FileNotFoundError: If no files match the pattern.
    """
    files = sorted(Path(weather_glob).parent.glob(Path(weather_glob).name))
    if not files:
        raise FileNotFoundError(f"No weather files found for pattern: {weather_glob}")
    logger.info("Found %d weather files matching %s", len(files), weather_glob)
    return files


def weather_ds_to_dataframe(ds: xr.Dataset, time_coord: str = "time") -> pd.DataFrame:
    """Convert an xarray Dataset to a flat DataFrame with standardised column names.

    Args:
        ds: Source xarray Dataset.
        time_coord: Name of the time coordinate.

    Returns:
        DataFrame with timestamp, era5_lat, era5_lon, and weather_* columns.

    Raises:
        ValueError: If time_coord is absent from the dataset.
    """
    if time_coord not in ds.coords:
        raise ValueError(f"Expected time coordinate '{time_coord}' in dataset")

    data_vars = list(ds.data_vars)
    df = ds[data_vars].to_dataframe().reset_index()

    rename: dict[str, str] = {time_coord: "timestamp"}
    rename.update({var: f"weather_{var}" for var in data_vars})

    lat_col = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon_col = next((c for c in ("longitude", "lon") if c in df.columns), None)
    if lat_col:
        rename[lat_col] = "era5_lat"
    if lon_col:
        rename[lon_col] = "era5_lon"

    df = df.rename(columns=rename)
    logger.info("Converted dataset with %d rows to DataFrame", len(df))
    return df


def build_weather_table(weather_glob: str, time_coord: str = "time") -> pd.DataFrame:
    """Load and concatenate all ERA5 NetCDF files matching a glob pattern.

    Args:
        weather_glob: Glob pattern for ERA5 NetCDF files.
        time_coord: Name of the time coordinate in each file.

    Returns:
        Concatenated DataFrame across all matched files.
    """
    files = load_weather_files(weather_glob)
    frames: list[pd.DataFrame] = []
    for path in files:
        logger.info("Loading weather file %s", path)
        with xr.open_dataset(path) as ds:
            frames.append(weather_ds_to_dataframe(ds, time_coord=time_coord))
    weather_df = pd.concat(frames, ignore_index=True)
    logger.info("Built weather table with %d rows from %d files", len(weather_df), len(files))
    return weather_df


def merge_weather_with_mapping(
    weather_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join weather data with site mapping on ERA5 grid coordinates.

    Args:
        weather_df: ERA5 weather DataFrame with era5_lat and era5_lon columns.
        mapping_df: Site mapping DataFrame with era5_lat and era5_lon columns.

    Returns:
        Merged DataFrame.

    Raises:
        ValueError: If join columns are absent from either DataFrame.
    """
    join_cols = ["era5_lat", "era5_lon"]
    for col in join_cols:
        if col not in weather_df.columns or col not in mapping_df.columns:
            raise ValueError(f"Expected column '{col}' in both weather and mapping data")

    merged = weather_df.merge(mapping_df, on=join_cols, how="inner")
    if "location" not in merged.columns:
        logger.warning("Result of merging weather and mapping has no 'location' column")
    logger.info("Merged weather with mapping, resulting rows: %d", len(merged))
    return merged


def merge_power_with_weather(
    power_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join power demand with weather data on location and timestamp.

    Args:
        power_df: Power demand DataFrame with location and timestamp columns.
        weather_df: Weather DataFrame with location and timestamp columns.

    Returns:
        Merged DataFrame.

    Raises:
        ValueError: If required join columns are absent from either DataFrame.
    """
    for col in ("timestamp", "location"):
        if col not in power_df.columns:
            raise ValueError(f"Power data must have '{col}' column")
        if col not in weather_df.columns:
            raise ValueError(f"Weather data must have '{col}' column")

    merged = power_df.merge(weather_df, on=["location", "timestamp"], how="inner")
    logger.info(
        "Merged power with weather on location and timestamp, resulting rows: %d",
        len(merged),
    )
    return merged
