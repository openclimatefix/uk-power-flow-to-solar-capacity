import os
import logging
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd
import geopandas as gpd

from .utils import (
    load_preprocess_config,
    analyze_directory,
    extract_zip,
    analyze_top_level_folders,
    find_data_files,
    jaccard_token_overlap,
)


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CFG = load_preprocess_config()

PATHS = CFG.paths
PROCESSING = CFG.processing
MATCHING = CFG.matching

BASE_DATA_DIR = PATHS.base_data_dir
ZIP_FILENAME = PATHS.zip_filename
EXTRACT_DIRNAME = PATHS.extract_dirname

AGG_CSV = PATHS.agg_csv
GEOJSON = PATHS.geojson
OUT_MATCH = PATHS.out_match
OUT_UNMATCHED = PATHS.out_unmatched

COMBINED_FULL_CSV = PATHS.combined_full_csv
COMBINED_POWER_ONLY_CSV = PATHS.combined_power_only_csv
COMBINED_REDUCED_CSV = PATHS.combined_reduced_csv
COMBINED_FILLED_POWER_CSV = PATHS.combined_filled_power_csv
COMBINED_AGG_LOCATION_CSV = PATHS.combined_aggregated_location_csv
AGG_COORDS_CSV = PATHS.aggregated_with_coords_csv

FUZZY_THRESHOLD = float(PROCESSING.fuzzy_threshold)


def combine_all_hh_files(
    base_path: str | None = None,
    output_path: str | None = None,
    sample_run: bool | None = None,
) -> tuple[pd.DataFrame | None, dict | None]:
    if base_path is None:
        base_path = os.path.join(BASE_DATA_DIR, EXTRACT_DIRNAME, "Primary Transformers")
    if output_path is None:
        output_path = os.path.join(BASE_DATA_DIR, "combined_ukpn_hh_data.csv")
    if sample_run is None:
        sample_run = bool(PROCESSING.sample_run)

    logger.info("Combining HH files from %s", base_path)

    hh_files: list[str] = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith("_HH_data.csv"):
                hh_files.append(os.path.join(root, file))

    if sample_run:
        limit = int(PROCESSING.sample_file_limit)
        hh_files = hh_files[:limit]
        logger.info("Sample run enabled, using first %d files", len(hh_files))

    combined_df = pd.DataFrame()
    total_rows = 0

    for file_path in hh_files:
        try:
            df = pd.read_csv(file_path)
            required_cols = ["tx_id", "hh", "kva"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning("Missing columns %s in file %s", missing_cols, file_path)
                continue
            df["hh"] = pd.to_datetime(df["hh"], errors="coerce")
            before_rows = len(df)
            df = df.dropna(subset=["hh"])
            after_rows = len(df)
            dropped = before_rows - after_rows
            if dropped > 0:
                logger.info("Dropped %d rows with invalid hh in %s", dropped, file_path)
            total_rows += after_rows
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)

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
    base_path: str | None = None,
    output_path: str | None = None,
    chunk_size: int | None = None,
) -> tuple[str, dict]:
    if base_path is None:
        base_path = os.path.join(BASE_DATA_DIR, EXTRACT_DIRNAME, "Primary Transformers")
    if output_path is None:
        output_path = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_streaming)

    logger.info("Streaming combination of HH files from %s", base_path)

    hh_files: list[str] = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith("_HH_data.csv"):
                hh_files.append(os.path.join(root, file))

    if os.path.exists(output_path):
        os.remove(output_path)

    total_rows_written = 0
    unique_tx_ids: set[str] = set()
    first_chunk_written = False

    for file_path in hh_files:
        try:
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
                header = not first_chunk_written
                chunk.to_csv(output_path, mode=mode, index=False, header=header)
                first_chunk_written = True
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)

    logger.info("Streaming combination complete: %d rows, %d transformers", total_rows_written, len(unique_tx_ids))
    return output_path, {
        "num_files": len(hh_files),
        "total_rows": total_rows_written,
        "unique_tx_ids": len(unique_tx_ids),
    }


def quick_file_check(filepath: str | None = None, chunk_size: int | None = None) -> None:
    if filepath is None:
        filepath = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_quick_check)
    if not os.path.exists(filepath):
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
    input_file: str | None = None,
    chunk_size: int | None = None,
) -> tuple[str, str]:
    if input_file is None:
        input_file = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_optimized)

    power_only_path = COMBINED_POWER_ONLY_CSV
    reduced_path = COMBINED_REDUCED_CSV

    if os.path.exists(power_only_path):
        os.remove(power_only_path)
    if os.path.exists(reduced_path):
        os.remove(reduced_path)

    for chunk in pd.read_csv(input_file, chunksize=chunk_size, parse_dates=["hh"]):
        power_chunk = chunk[["tx_id", "hh", "kva"]].copy()
        power_chunk["hh"] = pd.to_datetime(power_chunk["hh"], errors="coerce")
        power_chunk["kva"] = pd.to_numeric(power_chunk["kva"], errors="coerce")
        header_power = not os.path.exists(power_only_path)
        power_chunk.to_csv(power_only_path, mode="a", header=header_power, index=False)

        reduced_chunk = chunk.copy()
        numeric_cols = reduced_chunk.select_dtypes(include=["int64", "float64"]).columns
        reduced_chunk[numeric_cols] = reduced_chunk[numeric_cols].apply(pd.to_numeric, downcast="float")
        header_reduced = not os.path.exists(reduced_path)
        reduced_chunk.to_csv(reduced_path, mode="a", header=header_reduced, index=False)

    logger.info("Optimized versions created: %s, %s", power_only_path, reduced_path)
    return power_only_path, reduced_path


def check_combined_file(filepath: str | None = None, chunk_size: int | None = None) -> None:
    if filepath is None:
        filepath = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_quick_check)
    if not os.path.exists(filepath):
        logger.warning("File %s not found", filepath)
        return

    total_rows = 0
    unique_tx_ids: set[str] = set()
    min_date: datetime | None = None
    max_date: datetime | None = None

    for chunk_idx, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size, parse_dates=["hh"]), start=1):
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


def check_file_variants() -> None:
    files_to_check = [
        COMBINED_FULL_CSV,
        COMBINED_POWER_ONLY_CSV,
        COMBINED_REDUCED_CSV,
    ]
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            logger.info("File %s does not exist", filepath)
            continue
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        logger.info("File %s exists, size %.2f MB", filepath, file_size)
        try:
            pd.read_csv(filepath, nrows=5)
        except Exception as e:
            logger.error("Error reading file %s: %s", filepath, e)


def count_unique_tx_ids(filepath: str | None = None, chunk_size: int | None = None) -> set[str]:
    if filepath is None:
        filepath = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_count_unique)
    unique_tx_ids: set[str] = set()
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        unique_tx_ids.update(chunk["tx_id"].unique())
    logger.info("Unique tx_id count: %d", len(unique_tx_ids))
    return unique_tx_ids


def get_all_unique_tx_ids_alphabetically(filepath: str | None = None, chunk_size: int | None = None) -> list[str]:
    unique_tx_ids = count_unique_tx_ids(filepath=filepath, chunk_size=chunk_size)
    unique_tx_ids_sorted = sorted(unique_tx_ids)
    logger.info("Unique tx_id count: %d", len(unique_tx_ids_sorted))
    return unique_tx_ids_sorted


def rank_all_transformers_by_completeness(
    filepath: str | None = None,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    if filepath is None:
        filepath = COMBINED_FULL_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_rank)

    total_counts: dict[str, int] = defaultdict(int)
    non_nan_counts: dict[str, int] = defaultdict(int)

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=["tx_id", "kva"]):
        for tx_id, group in chunk.groupby("tx_id"):
            total_counts[tx_id] += len(group)
            non_nan_counts[tx_id] += group["kva"].notna().sum()

    summary_data = []
    for tx_id in total_counts:
        total = total_counts[tx_id]
        non_nan = non_nan_counts[tx_id]
        fraction = non_nan / total if total > 0 else 0.0
        summary_data.append((tx_id, total, non_nan, fraction))

    summary_df = pd.DataFrame(summary_data, columns=["tx_id", "total_points", "non_nan_points", "fraction_non_nan"])
    summary_df = summary_df.sort_values(by=["fraction_non_nan", "non_nan_points"], ascending=[False, False]).reset_index(drop=True)
    output_path = os.path.join(BASE_DATA_DIR, "transformer_completeness_summary.csv")
    summary_df.to_csv(output_path, index=False)
    logger.info("Completeness summary saved to %s", output_path)
    return summary_df


def calculate_and_fill_power(
    input_filepath: str | None = None,
    output_filepath: str | None = None,
    transformer_id: str | None = None,
    chunk_size: int | None = None,
    power_factor: float | None = None,
) -> pd.DataFrame | None:
    if input_filepath is None:
        input_filepath = COMBINED_FULL_CSV
    if output_filepath is None:
        output_filepath = COMBINED_FILLED_POWER_CSV
    if chunk_size is None:
        chunk_size = int(PROCESSING.chunk_size_fill_power)
    if power_factor is None:
        power_factor = float(PROCESSING.power_factor)
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
        chunk_filtered["active_power_kW"] = chunk_filtered["active_power_kW"].fillna(method="ffill")
        processed_chunks.append(chunk_filtered)

    if not processed_chunks:
        logger.warning("No data found for transformer %s", transformer_id)
        return None

    result_df = pd.concat(processed_chunks, ignore_index=True)
    result_df = result_df.sort_values(by="hh")
    result_df.to_csv(output_filepath, index=False)
    logger.info("Active power for transformer %s written to %s", transformer_id, output_filepath)
    return result_df


def create_location_aggregated_dataset(
    input_file: str | None = None,
    output_file: str | None = None,
    chunksize: int | None = None,
) -> None:
    if input_file is None:
        input_file = COMBINED_FILLED_POWER_CSV
    if output_file is None:
        output_file = COMBINED_AGG_LOCATION_CSV
    if chunksize is None:
        chunksize = int(PROCESSING.chunk_size_aggregate_location)

    if os.path.exists(output_file):
        os.remove(output_file)

    location_cols = ["tx_id"]
    total_rows_output = 0

    for chunk in pd.read_csv(input_file, chunksize=chunksize, parse_dates=["hh"]):
        required_cols = set(location_cols + ["hh", "active_power_kW"])
        if not required_cols.issubset(chunk.columns):
            continue
        group_cols = location_cols + ["hh"]
        agg = chunk.groupby(group_cols, as_index=False)["active_power_kW"].sum()
        header = not os.path.exists(output_file)
        agg.to_csv(output_file, index=False, mode="a", header=header)
        total_rows_output += len(agg)

    logger.info("Location-aggregated dataset created at %s with %d rows", output_file, total_rows_output)


def inspect_geojson_file(filepath: str | None = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = GEOJSON
    gdf = gpd.read_file(filepath)
    logger.info("GeoJSON loaded with %d features", len(gdf))
    return gdf


def load_agg_data(filepath: str | None = None) -> pd.DataFrame:
    if filepath is None:
        filepath = AGG_CSV
    df = pd.read_csv(filepath)
    if "tx_id" not in df.columns:
        raise ValueError("Expected tx_id in aggregated dataset")
    df["location_name"] = df["tx_id"].astype(str)
    logger.info("Aggregated data loaded from %s with %d rows", filepath, len(df))
    return df


def load_geojson_data(filepath: str | None = None) -> pd.DataFrame:
    if filepath is None:
        filepath = GEOJSON
    gdf = gpd.read_file(filepath)

    candidate_name_cols = list(MATCHING.geojson_candidate_name_cols)
    name_col = None
    for c in candidate_name_cols:
        if c in gdf.columns:
            name_col = c
            break

    if name_col is None:
        gdf["geo_name"] = gdf.index.astype(str)
    else:
        gdf["geo_name"] = gdf[name_col].astype(str)

    centroids = gdf.geometry.centroid
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y

    out = gdf[["geo_name", "longitude", "latitude"]].copy()
    logger.info("GeoJSON subset prepared from %s with %d rows", filepath, len(out))
    return out


def match_locations_to_geo(
    df_agg: pd.DataFrame,
    df_geo: pd.DataFrame,
    fuzzy_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fuzzy_threshold is None:
        fuzzy_threshold = float(FUZZY_THRESHOLD)

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
            matches.append(
                {
                    "location_name": loc,
                    "matched_geo_name": best_row["geo_name"],
                    "score": best_score,
                    "longitude": best_row["longitude"],
                    "latitude": best_row["latitude"],
                }
            )
        else:
            unmatched.append({"location_name": loc})

    df_matches = pd.DataFrame(matches)
    df_unmatched = pd.DataFrame(unmatched)
    logger.info("Location matching complete: %d matched, %d unmatched", len(df_matches), len(df_unmatched))
    return df_matches, df_unmatched


def attach_coordinates_to_agg(df_agg: pd.DataFrame, df_matches: pd.DataFrame) -> pd.DataFrame:
    df_matches_reduced = df_matches[["location_name", "longitude", "latitude"]].drop_duplicates()
    df_merged = df_agg.merge(df_matches_reduced, on="location_name", how="left")
    logger.info("Coordinates merged into aggregated data, resulting rows %d", len(df_merged))
    return df_merged


def main_power_location_matching() -> None:
    if not os.path.exists(AGG_CSV):
        logger.error("Aggregated CSV not found at %s", AGG_CSV)
        return
    if not os.path.exists(GEOJSON):
        logger.error("GeoJSON file not found at %s", GEOJSON)
        return

    df_agg = load_agg_data(AGG_CSV)
    df_geo = load_geojson_data(GEOJSON)
    df_matches, df_unmatched = match_locations_to_geo(df_agg, df_geo, FUZZY_THRESHOLD)
    df_matches.to_csv(OUT_MATCH, index=False)
    df_unmatched.to_csv(OUT_UNMATCHED, index=False)
    df_with_coords = attach_coordinates_to_agg(df_agg, df_matches)
    df_with_coords.to_csv(AGG_COORDS_CSV, index=False)
    logger.info("Power location matching complete, outputs written")


def main() -> None:
    logger.info("Starting preprocessing pipeline")

    analyze_directory(BASE_DATA_DIR)
    zip_path = os.path.join(BASE_DATA_DIR, ZIP_FILENAME)
    extract_to = os.path.join(BASE_DATA_DIR, EXTRACT_DIRNAME)
    extract_zip(zip_path, extract_to)
    analyze_top_level_folders(extract_to)
    find_data_files(extract_to)

    combine_hh_files_streaming()
    quick_file_check()
    create_optimized_versions_streaming()
    check_file_variants()
    rank_all_transformers_by_completeness()

    create_location_aggregated_dataset()
    main_power_location_matching()

    logger.info("Preprocessing pipeline completed")


if __name__ == "__main__":
    main()
