"""Preprocessing pipeline for UKPN half-hourly power demand data."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas
import hydra
import pandas as pd
from omegaconf import DictConfig

from process.utils import (
    analyze_directory,
    analyze_top_level_folders,
    extract_zip,
    find_data_files,
    jaccard_token_overlap,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def combine_all_hh_files(
    cfg: DictConfig,
    base_path: str | None = None,
    output_path: str | None = None,
    sample_run: bool | None = None,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Combine all half-hourly CSV files into a single DataFrame.

    Args:
        cfg: Full Hydra config.
        base_path: Directory to walk for _HH_data.csv files.
        output_path: Destination CSV path.
        sample_run: If True, limit to the first sample_file_limit files.

    Returns:
        Tuple of (combined DataFrame, summary dict), or (None, None) if no data.
    """
    paths = cfg.paths
    processing = cfg.processing

    if base_path is None:
        base_path = str(Path(paths.base_data_dir) / paths.extract_dirname / "Primary Transformers")
    if output_path is None:
        output_path = str(Path(paths.base_data_dir) / "combined_ukpn_hh_data.csv")
    if sample_run is None:
        sample_run = bool(processing.sample_run)

    logger.info("Combining HH files from %s", base_path)

    hh_files: list[str] = [str(p) for p in Path(base_path).rglob("*_HH_data.csv")]

    if sample_run:
        limit = int(processing.sample_file_limit)
        hh_files = hh_files[:limit]
        logger.info("Sample run enabled, using first %d files", len(hh_files))

    combined_df = pd.DataFrame()

    for file_path in hh_files:
        df = pd.read_csv(file_path)
        required_cols = ["tx_id", "hh", "kva"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning("Missing columns %s in file %s", missing_cols, file_path)
            continue
        df["hh"] = pd.to_datetime(df["hh"], errors="coerce")
        before_rows = len(df)
        df = df.dropna(subset=["hh"])
        dropped = before_rows - len(df)
        if dropped > 0:
            logger.info("Dropped %d rows with invalid hh in %s", dropped, file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        logger.warning("No data combined")
        return None, None

    combined_df = combined_df.sort_values(by=["tx_id", "hh"]).reset_index(drop=True)
    combined_df.to_csv(output_path, index=False)

    unique_tx_ids = combined_df["tx_id"].nunique()
    date_range = (combined_df["hh"].min(), combined_df["hh"].max())

    logger.info(
        "Combined %d rows from %d files, %d transformers, date range %s to %s",
        len(combined_df),
        len(hh_files),
        unique_tx_ids,
        date_range[0],
        date_range[1],
    )
    return combined_df, {
        "num_files": len(hh_files),
        "total_rows": len(combined_df),
        "unique_tx_ids": unique_tx_ids,
        "date_range": date_range,
    }


def combine_hh_files_streaming(
    cfg: DictConfig,
    base_path: str | None = None,
    output_path: str | None = None,
    chunk_size: int | None = None,
) -> tuple[str, dict]:
    """Stream-combine all half-hourly CSV files to avoid loading all into memory.

    Args:
        cfg: Full Hydra config.
        base_path: Directory to walk for _HH_data.csv files.
        output_path: Destination CSV path.
        chunk_size: Rows per chunk when reading each file.

    Returns:
        Tuple of (output_path, summary dict).
    """
    paths = cfg.paths
    processing = cfg.processing

    if base_path is None:
        base_path = str(Path(paths.base_data_dir) / paths.extract_dirname / "Primary Transformers")
    if output_path is None:
        output_path = str(paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(processing.chunk_size_streaming)

    logger.info("Streaming combination of HH files from %s", base_path)

    hh_files: list[str] = [str(p) for p in Path(base_path).rglob("*_HH_data.csv")]

    out = Path(output_path)
    if out.exists():
        out.unlink()

    total_rows_written = 0
    unique_tx_ids: set[str] = set()
    first_chunk_written = False

    for file_path in hh_files:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            required_cols = ["tx_id", "hh", "kva"]
            if not all(col in chunk.columns for col in required_cols):
                continue
            chunk["hh"] = pd.to_datetime(chunk["hh"], errors="coerce")
            chunk = chunk.dropna(subset=["hh"])
            unique_tx_ids.update(chunk["tx_id"].unique())
            total_rows_written += len(chunk)
            chunk = chunk.sort_values(by=["tx_id", "hh"])
            mode = "w" if not first_chunk_written else "a"
            chunk.to_csv(output_path, mode=mode, index=False, header=not first_chunk_written)
            first_chunk_written = True

    logger.info("Streaming: %d rows, %d transformers", total_rows_written, len(unique_tx_ids))
    return output_path, {
        "num_files": len(hh_files),
        "total_rows": total_rows_written,
        "unique_tx_ids": len(unique_tx_ids),
    }


def quick_file_check(
    cfg: DictConfig,
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> None:
    """Log summary statistics for a combined CSV file.

    Args:
        cfg: Full Hydra config.
        filepath: Path to the CSV to inspect.
        chunk_size: Rows per chunk when reading.
    """
    if filepath is None:
        filepath = str(cfg.paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(cfg.processing.chunk_size_quick_check)
    if not Path(filepath).exists():
        logger.warning("File %s not found", filepath)
        return

    total_rows = 0
    unique_tx_ids: set[str] = set()
    min_date: datetime | None = None
    max_date: datetime | None = None

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, parse_dates=["hh"]):
        total_rows += len(chunk)
        unique_tx_ids.update(chunk["tx_id"].unique())
        chunk_min = chunk["hh"].min()
        chunk_max = chunk["hh"].max()
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    logger.info(
        "File %s: %d rows, %d transformers, date range %s to %s",
        filepath,
        total_rows,
        len(unique_tx_ids),
        min_date,
        max_date,
    )


def create_optimized_versions_streaming(
    cfg: DictConfig,
    input_file: str | None = None,
    chunk_size: int | None = None,
) -> tuple[str, str]:
    """Create power-only and type-reduced versions of the combined CSV.

    Args:
        cfg: Full Hydra config.
        input_file: Source CSV path.
        chunk_size: Rows per chunk when reading.

    Returns:
        Tuple of (power_only_path, reduced_path).
    """
    paths = cfg.paths
    processing = cfg.processing

    if input_file is None:
        input_file = str(paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(processing.chunk_size_optimized)

    power_only_path = str(paths.combined_power_only_csv)
    reduced_path = str(paths.combined_reduced_csv)

    for p in (power_only_path, reduced_path):
        if Path(p).exists():
            Path(p).unlink()

    for chunk in pd.read_csv(input_file, chunksize=chunk_size, parse_dates=["hh"]):
        power_chunk = chunk[["tx_id", "hh", "kva"]].copy()
        power_chunk["hh"] = pd.to_datetime(power_chunk["hh"], errors="coerce")
        power_chunk["kva"] = pd.to_numeric(power_chunk["kva"], errors="coerce")
        header_power = not Path(power_only_path).exists()
        power_chunk.to_csv(power_only_path, mode="a", header=header_power, index=False)

        reduced_chunk = chunk.copy()
        numeric_cols = reduced_chunk.select_dtypes(include=["int64", "float64"]).columns
        reduced_chunk[numeric_cols] = reduced_chunk[numeric_cols].apply(
            pd.to_numeric, downcast="float"
        )
        header_reduced = not Path(reduced_path).exists()
        reduced_chunk.to_csv(reduced_path, mode="a", header=header_reduced, index=False)

    logger.info("Optimized versions created: %s, %s", power_only_path, reduced_path)
    return power_only_path, reduced_path


def check_combined_file(
    cfg: DictConfig,
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> None:
    """Log detailed diagnostics for a combined CSV including missing value counts.

    Args:
        cfg: Full Hydra config.
        filepath: Path to the CSV to inspect.
        chunk_size: Rows per chunk when reading.
    """
    if filepath is None:
        filepath = str(cfg.paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(cfg.processing.chunk_size_quick_check)
    if not Path(filepath).exists():
        logger.warning("File %s not found", filepath)
        return

    total_rows = 0
    unique_tx_ids: set[str] = set()
    min_date: datetime | None = None
    max_date: datetime | None = None

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, parse_dates=["hh"]):
        total_rows += len(chunk)
        unique_tx_ids.update(chunk["tx_id"].unique())
        chunk_min = chunk["hh"].min()
        chunk_max = chunk["hh"].max()
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    sample_chunk = next(pd.read_csv(filepath, chunksize=chunk_size, parse_dates=["hh"]))
    missing_kva = sample_chunk["kva"].isna().sum()

    logger.info(
        "Combined file %s: %d rows, %d transformers, date range %s to %s, missing kva in sample %d",
        filepath,
        total_rows,
        len(unique_tx_ids),
        min_date,
        max_date,
        missing_kva,
    )


def check_file_variants(cfg: DictConfig) -> None:
    """Log existence and size of each known combined CSV variant.

    Args:
        cfg: Full Hydra config.
    """
    paths = cfg.paths
    files_to_check = [
        str(paths.combined_full_csv),
        str(paths.combined_power_only_csv),
        str(paths.combined_reduced_csv),
    ]
    for filepath in files_to_check:
        p = Path(filepath)
        if not p.exists():
            logger.info("File %s does not exist", filepath)
            continue
        file_size = p.stat().st_size / (1024 * 1024)
        logger.info("File %s exists, size %.2f MB", filepath, file_size)
        pd.read_csv(filepath, nrows=5)


def count_unique_tx_ids(
    cfg: DictConfig,
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> set[str]:
    """Count unique transformer IDs across a CSV file.

    Args:
        cfg: Full Hydra config.
        filepath: Source CSV path.
        chunk_size: Rows per chunk when reading.

    Returns:
        Set of unique tx_id strings.
    """
    if filepath is None:
        filepath = str(cfg.paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(cfg.processing.chunk_size_count_unique)

    unique_tx_ids: set[str] = set()
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        unique_tx_ids.update(chunk["tx_id"].unique())

    logger.info("Unique tx_id count: %d", len(unique_tx_ids))
    return unique_tx_ids


def get_all_unique_tx_ids_alphabetically(
    cfg: DictConfig,
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> list[str]:
    """Return all unique transformer IDs sorted alphabetically.

    Args:
        cfg: Full Hydra config.
        filepath: Source CSV path.
        chunk_size: Rows per chunk when reading.

    Returns:
        Sorted list of tx_id strings.
    """
    unique_tx_ids = count_unique_tx_ids(cfg, filepath=filepath, chunk_size=chunk_size)
    unique_tx_ids_sorted = sorted(unique_tx_ids)
    logger.info("Unique tx_id count: %d", len(unique_tx_ids_sorted))
    return unique_tx_ids_sorted


def rank_all_transformers_by_completeness(
    cfg: DictConfig,
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Rank transformers by fraction of non-null kva readings.

    Args:
        cfg: Full Hydra config.
        filepath: Source CSV path.
        chunk_size: Rows per chunk when reading.

    Returns:
        DataFrame sorted by completeness descending.
    """
    if filepath is None:
        filepath = str(cfg.paths.combined_full_csv)
    if chunk_size is None:
        chunk_size = int(cfg.processing.chunk_size_rank)

    total_counts: dict[str, int] = defaultdict(int)
    non_nan_counts: dict[str, int] = defaultdict(int)

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=["tx_id", "kva"]):
        for tx_id, group in chunk.groupby("tx_id"):
            total_counts[tx_id] += len(group)
            non_nan_counts[tx_id] += int(group["kva"].notna().sum())

    summary_data = [
        (
            tx_id,
            total_counts[tx_id],
            non_nan_counts[tx_id],
            non_nan_counts[tx_id] / total_counts[tx_id] if total_counts[tx_id] > 0 else 0.0,
        )
        for tx_id in total_counts
    ]

    summary_df = pd.DataFrame(
        summary_data,
        columns=["tx_id", "total_points", "non_nan_points", "fraction_non_nan"],
    )
    summary_df = summary_df.sort_values(
        by=["fraction_non_nan", "non_nan_points"], ascending=[False, False]
    ).reset_index(drop=True)

    output_path = Path(cfg.paths.base_data_dir) / "transformer_completeness_summary.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info("Completeness summary saved to %s", output_path)
    return summary_df


def calculate_and_fill_power(
    cfg: DictConfig,
    input_filepath: str | None = None,
    output_filepath: str | None = None,
    transformer_id: str | None = None,
    chunk_size: int | None = None,
    power_factor: float | None = None,
) -> pd.DataFrame | None:
    """Compute active power for one transformer and forward-fill gaps.

    Args:
        cfg: Full Hydra config.
        input_filepath: Source CSV path.
        output_filepath: Destination CSV path.
        transformer_id: tx_id to process.
        chunk_size: Rows per chunk when reading.
        power_factor: Multiplier applied to kva to derive active power.

    Returns:
        Processed DataFrame, or None if no data found for transformer_id.
    """
    if input_filepath is None:
        input_filepath = str(cfg.paths.combined_full_csv)
    if output_filepath is None:
        output_filepath = str(cfg.paths.combined_filled_power_csv)
    if chunk_size is None:
        chunk_size = int(cfg.processing.chunk_size_fill_power)
    if power_factor is None:
        power_factor = float(cfg.processing.power_factor)
    if transformer_id is None:
        logger.warning("No transformer_id provided")
        return None

    processed_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(input_filepath, chunksize=chunk_size):
        chunk_filtered = chunk[chunk["tx_id"] == transformer_id].copy()
        if chunk_filtered.empty:
            continue
        chunk_filtered["hh"] = pd.to_datetime(chunk_filtered["hh"], errors="coerce")
        chunk_filtered = chunk_filtered.dropna(subset=["hh"])
        chunk_filtered["active_power_kW"] = chunk_filtered["kva"] * power_factor
        chunk_filtered = chunk_filtered.sort_values(by="hh")
        chunk_filtered["active_power_kW"] = chunk_filtered["active_power_kW"].ffill()
        processed_chunks.append(chunk_filtered)

    if not processed_chunks:
        logger.warning("No data found for transformer %s", transformer_id)
        return None

    result_df = pd.concat(processed_chunks, ignore_index=True).sort_values(by="hh")
    result_df.to_csv(output_filepath, index=False)
    logger.info("Active power for transformer %s written to %s", transformer_id, output_filepath)
    return result_df


def create_location_aggregated_dataset(
    cfg: DictConfig,
    input_file: str | None = None,
    output_file: str | None = None,
    chunksize: int | None = None,
) -> None:
    """Aggregate active power by location and timestamp.

    Args:
        cfg: Full Hydra config.
        input_file: Source CSV path.
        output_file: Destination CSV path.
        chunksize: Rows per chunk when reading.
    """
    if input_file is None:
        input_file = str(cfg.paths.combined_filled_power_csv)
    if output_file is None:
        output_file = str(cfg.paths.combined_aggregated_location_csv)
    if chunksize is None:
        chunksize = int(cfg.processing.chunk_size_aggregate_location)

    out = Path(output_file)
    if out.exists():
        out.unlink()

    location_cols = ["tx_id"]
    total_rows_output = 0

    for chunk in pd.read_csv(input_file, chunksize=chunksize, parse_dates=["hh"]):
        required_cols = {*location_cols, "hh", "active_power_kW"}
        if not required_cols.issubset(chunk.columns):
            continue
        group_cols = [*location_cols, "hh"]
        agg = chunk.groupby(group_cols, as_index=False)["active_power_kW"].sum()
        agg.to_csv(output_file, index=False, mode="a", header=not out.exists())
        total_rows_output += len(agg)

    logger.info("Aggregated dataset created at %s with %d rows", output_file, total_rows_output)


def inspect_geojson_file(
    cfg: DictConfig,
    filepath: str | None = None,
) -> geopandas.GeoDataFrame:
    """Load and log summary of a GeoJSON file.

    Args:
        cfg: Full Hydra config.
        filepath: Path to GeoJSON file.

    Returns:
        Loaded GeoDataFrame.
    """
    if filepath is None:
        filepath = str(cfg.paths.geojson)
    gdf = geopandas.read_file(filepath)
    logger.info("GeoJSON loaded with %d features", len(gdf))
    return gdf


def load_agg_data(
    cfg: DictConfig,
    filepath: str | None = None,
) -> pd.DataFrame:
    """Load the location-aggregated CSV and add a location_name column.

    Args:
        cfg: Full Hydra config.
        filepath: Path to aggregated CSV.

    Returns:
        DataFrame with location_name column added.

    Raises:
        ValueError: If tx_id column is absent.
    """
    if filepath is None:
        filepath = str(cfg.paths.agg_csv)
    df = pd.read_csv(filepath)
    if "tx_id" not in df.columns:
        raise ValueError("Expected tx_id in aggregated dataset")
    df["location_name"] = df["tx_id"].astype(str)
    logger.info("Aggregated data loaded from %s with %d rows", filepath, len(df))
    return df


def load_geojson_data(
    cfg: DictConfig,
    filepath: str | None = None,
) -> pd.DataFrame:
    """Load GeoJSON and extract centroid coordinates with a normalised name column.

    Args:
        cfg: Full Hydra config.
        filepath: Path to GeoJSON file.

    Returns:
        DataFrame with columns (geo_name, longitude, latitude).
    """
    if filepath is None:
        filepath = str(cfg.paths.geojson)
    gdf = geopandas.read_file(filepath)

    candidate_name_cols = list(cfg.matching.geojson_candidate_name_cols)
    name_col = next((c for c in candidate_name_cols if c in gdf.columns), None)

    gdf["geo_name"] = gdf[name_col].astype(str) if name_col else gdf.index.astype(str)
    centroids = gdf.geometry.centroid
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y

    out = gdf[["geo_name", "longitude", "latitude"]].copy()
    logger.info("GeoJSON subset prepared from %s with %d rows", filepath, len(out))
    return out


def match_locations_to_geo(
    cfg: DictConfig,
    df_agg: pd.DataFrame,
    df_geo: pd.DataFrame,
    fuzzy_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuzzy-match aggregated location names to GeoJSON features.

    Args:
        cfg: Full Hydra config.
        df_agg: Aggregated power DataFrame with location_name column.
        df_geo: GeoJSON-derived DataFrame with geo_name, longitude, latitude columns.
        fuzzy_threshold: Minimum Jaccard score to accept a match.

    Returns:
        Tuple of (matched DataFrame, unmatched DataFrame).
    """
    if fuzzy_threshold is None:
        fuzzy_threshold = float(cfg.processing.fuzzy_threshold)

    unique_locs = df_agg["location_name"].unique()
    matches = []
    unmatched = []

    for loc in unique_locs:
        best_score = 0.0
        best_row = None
        for _, row in df_geo.iterrows():
            score = jaccard_token_overlap(loc, row["geo_name"])
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is not None and best_score >= fuzzy_threshold:
            matches.append({
                "location_name": loc,
                "matched_geo_name": best_row["geo_name"],
                "score": best_score,
                "longitude": best_row["longitude"],
                "latitude": best_row["latitude"],
            })
        else:
            unmatched.append({"location_name": loc})

    df_matches = pd.DataFrame(matches)
    df_unmatched = pd.DataFrame(unmatched)
    logger.info(
        "Location matching complete: %d matched, %d unmatched",
        len(df_matches),
        len(df_unmatched),
    )
    return df_matches, df_unmatched


def attach_coordinates_to_agg(
    df_agg: pd.DataFrame,
    df_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join match coordinates onto the aggregated power DataFrame.

    Args:
        df_agg: Aggregated power DataFrame.
        df_matches: Match results with location_name, longitude, latitude.

    Returns:
        Merged DataFrame with coordinates attached.
    """
    df_matches_reduced = df_matches[["location_name", "longitude", "latitude"]].drop_duplicates()
    df_merged = df_agg.merge(df_matches_reduced, on="location_name", how="left")
    logger.info("Coordinates merged into aggregated data, resulting rows %d", len(df_merged))
    return df_merged


def main_power_location_matching(cfg: DictConfig) -> None:
    """Run the full location-to-GeoJSON matching and write results.

    Args:
        cfg: Full Hydra config.

    """
    paths = cfg.paths

    agg_csv = Path(paths.agg_csv)
    geojson = Path(paths.geojson)

    if not agg_csv.exists():
        logger.error("Aggregated CSV not found at %s", agg_csv)
        return
    if not geojson.exists():
        logger.error("GeoJSON file not found at %s", geojson)
        return

    df_agg = load_agg_data(cfg)
    df_geo = load_geojson_data(cfg)
    df_matches, df_unmatched = match_locations_to_geo(cfg, df_agg, df_geo)
    df_matches.to_csv(paths.out_match, index=False)
    df_unmatched.to_csv(paths.out_unmatched, index=False)
    df_with_coords = attach_coordinates_to_agg(df_agg, df_matches)
    df_with_coords.to_csv(paths.aggregated_with_coords_csv, index=False)
    logger.info("Power location matching complete, outputs written")


@hydra.main(version_base=None, config_path="../../configs/process", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the full preprocessing pipeline.

    Args:
        cfg: Hydra config injected automatically.
    """
    logger.info("Starting preprocessing pipeline")

    paths = cfg.paths
    base_data_dir = Path(paths.base_data_dir)
    zip_path = base_data_dir / paths.zip_filename
    extract_to = base_data_dir / paths.extract_dirname

    analyze_directory(str(base_data_dir))
    extract_zip(str(zip_path), str(extract_to))
    analyze_top_level_folders(str(extract_to))
    find_data_files(str(extract_to))

    combine_hh_files_streaming(cfg)
    quick_file_check(cfg)
    create_optimized_versions_streaming(cfg)
    check_file_variants(cfg)
    rank_all_transformers_by_completeness(cfg)

    create_location_aggregated_dataset(cfg)
    main_power_location_matching(cfg)

    logger.info("Preprocessing pipeline completed")


if __name__ == "__main__":
    main()
