import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from pyproj import Transformer
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TFT_INPUT_DATA_PATH = "/home/felix/output/tft_ready_data.parquet"
OUTPUT_DIRECTORY = "/home/felix/output/"
DATA_DIRECTORY = "/home/felix/post-fe-data/"
PASSIVE_PV_DATA_PATH = "/home/felix/passive_pv_data/30_minutely/"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


logger.warning("Loading main simulation data from: %s", TFT_INPUT_DATA_PATH)
try:
    df = pd.read_parquet(TFT_INPUT_DATA_PATH)
    logger.warning("Main data loaded. Shape: %s", str(df.shape))
except FileNotFoundError:
    logger.error("Data file not found at '%s'", TFT_INPUT_DATA_PATH)
    raise SystemExit(1)


if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    logger.warning("Full date range: %s to %s", min_date, max_date)
    df_sorted = df.sort_values(by=["location", "timestamp"]) 
    time_diffs = df_sorted.groupby("location")["timestamp"].diff()
    if not time_diffs.dropna().empty:
        resolution = time_diffs.mode()[0]
        logger.warning("Detected time resolution: %s", resolution)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    PV_MIN = pd.Timestamp("2021-01-01 00:30:00", tz="UTC")
    PV_MAX = pd.Timestamp("2025-04-02 00:00:00", tz="UTC")
    before_rows = len(df)
    df = df[(df["timestamp"] >= PV_MIN) & (df["timestamp"] <= PV_MAX)].copy()
    after_rows = len(df)
    logger.warning("Truncated rows: %d → %d", before_rows, after_rows)
else:
    logger.error("'timestamp' column not found")


all_locations = df["location"].unique().tolist()
logger.warning("Found %d unique locations", len(all_locations))
location_data = {location: df[df["location"] == location].copy() for location in all_locations}


all_coords_list = []
site_coordinates_df = pd.DataFrame()
try:
    parquet_files = [os.path.join(DATA_DIRECTORY, f) for f in os.listdir(DATA_DIRECTORY) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIRECTORY}")
    for file_path in parquet_files:
        temp_df = pd.read_parquet(file_path)
        if "latitude" in temp_df.columns and "longitude" in temp_df.columns:
            subset_df = temp_df[temp_df["location"].isin(all_locations)]
            if not subset_df.empty:
                all_coords_list.append(subset_df[["location", "latitude", "longitude"]])
    if all_coords_list:
        combined_coords_df = pd.concat(all_coords_list)
        site_coordinates_df = combined_coords_df.drop_duplicates().reset_index(drop=True)
        logger.warning("Coordinates found for %d locations", len(site_coordinates_df))
    else:
        logger.error("No locations were found in the coordinate files")
except Exception as e:
    logger.error("Error processing the Parquet directory for coordinates: %s", e)


def find_nearby_pv_systems(all_site_coords_df, radius_km=10):
    logger.warning("Downloading PV system metadata from Hugging Face")
    try:
        metadata_path = hf_hub_download(
            repo_id="openclimatefix/uk_pv",
            filename="metadata.csv",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        pv_metadata = pd.read_csv(metadata_path)
    except Exception as e:
        logger.error("Failed to download PV metadata: %s", e)
        return None
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    pv_metadata["x_osgb"], pv_metadata["y_osgb"] = transformer.transform(
        pv_metadata["longitude_rounded"].values,
        pv_metadata["latitude_rounded"].values,
    )
    training_x, training_y = transformer.transform(
        all_site_coords_df["longitude"].values,
        all_site_coords_df["latitude"].values,
    )
    training_points = np.c_[training_x, training_y]
    pv_points = pv_metadata[["x_osgb", "y_osgb"]].values
    pv_kdtree = cKDTree(pv_points)
    radius_m = radius_km * 1000
    nearby_indices = pv_kdtree.query_ball_point(training_points, r=radius_m)
    all_matches = []
    for i, site_row in all_site_coords_df.iterrows():
        site_name = site_row["location"]
        indices_for_this_site = nearby_indices[i]
        if indices_for_this_site:
            nearby_pv_systems = pv_metadata.iloc[indices_for_this_site].copy()
            site_point_osgb = training_points[i]
            distances = np.linalg.norm(nearby_pv_systems[["x_osgb", "y_osgb"]].values - site_point_osgb, axis=1)
            nearby_pv_systems["training_location"] = site_name
            nearby_pv_systems["distance_m"] = distances
            nearby_pv_systems["distance_km"] = distances / 1000
            nearby_pv_systems.rename(columns={"ss_id": "pv_ss_id", "kwp": "pv_capacity_kw"}, inplace=True)
            all_matches.append(nearby_pv_systems)
    if not all_matches:
        logger.error("No nearby PV systems found for any location")
        return None
    pv_mappings_df = pd.concat(all_matches, ignore_index=True)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    pv_matches_path = os.path.join(OUTPUT_DIRECTORY, "all_locations_pv_matches.csv")
    pv_mappings_df.to_csv(pv_matches_path, index=False)
    logger.warning("Saved matches to: %s", pv_matches_path)
    return pv_mappings_df


def dedupe_pv_to_nearest(pv_mappings_df):
    out = pv_mappings_df.copy()
    out["__rank__"] = out.groupby("pv_ss_id")["distance_m"].rank(method="first")
    out = out[out["__rank__"] == 1].drop(columns="__rank__")
    return out


def filter_mapping_to_present_ids(pv_mappings_df, passive_pv_path):
    parquet_files = glob.glob(os.path.join(passive_pv_path, "**", "*.parquet"), recursive=True)
    present_ids = set()
    for fp in parquet_files:
        try:
            tmp = pd.read_parquet(fp, columns=["ss_id"])
            present_ids.update(tmp["ss_id"].unique().tolist())
        except Exception:
            continue
    out = pv_mappings_df[pv_mappings_df["pv_ss_id"].isin(present_ids)].copy()
    return out


def estimate_capacities(passive_pv_path, target_ids):
    parquet_files = glob.glob(os.path.join(passive_pv_path, "**", "*.parquet"), recursive=True)
    pieces = []
    for fp in parquet_files:
        try:
            dfp = pd.read_parquet(fp, columns=["ss_id", "generation_Wh"])
            dfp = dfp[dfp["ss_id"].isin(target_ids)].copy()
            if dfp.empty:
                continue
            dfp["mw"] = dfp["generation_Wh"] / 1_000_000 / 0.5
            cap = dfp.groupby("ss_id")["mw"].quantile(0.95).rename("cap_mw_q95").reset_index()
            pieces.append(cap)
        except Exception:
            continue
    if not pieces:
        return pd.DataFrame(columns=["ss_id", "cap_kw_est"])
    out = pd.concat(pieces, ignore_index=True)
    out = out.groupby("ss_id")["cap_mw_q95"].max().reset_index()
    out["cap_kw_est"] = out["cap_mw_q95"] * 1000.0
    return out[["ss_id", "cap_kw_est"]]


if "site_coordinates_df" in locals() and not site_coordinates_df.empty:
    pv_mappings = find_nearby_pv_systems(site_coordinates_df)
    if pv_mappings is not None and not pv_mappings.empty:
        pv_mappings = dedupe_pv_to_nearest(pv_mappings)
        pv_mappings = filter_mapping_to_present_ids(pv_mappings, PASSIVE_PV_DATA_PATH)
        cap_tbl = estimate_capacities(PASSIVE_PV_DATA_PATH, set(pv_mappings["pv_ss_id"]))
        pv_mappings = pv_mappings.merge(cap_tbl, left_on="pv_ss_id", right_on="ss_id", how="left")
        pv_mappings.drop(columns=["ss_id"], inplace=True, errors="ignore")
        before = len(pv_mappings)
        pv_mappings = pv_mappings[(pv_mappings["cap_kw_est"].fillna(0) >= 0.5) | (pv_mappings["cap_kw_est"].isna())]
        after = len(pv_mappings)
        logger.warning("Pruned tiny PV IDs (<0.5 kW est): rows %d → %d", before, after)
    else:
        pv_mappings = None
else:
    logger.error("Cannot perform PV matching because location coordinates were not found")
    pv_mappings = None


def load_and_aggregate_passive_pv_data(pv_mappings_df, passive_pv_path):
    if pv_mappings_df is None or pv_mappings_df.empty:
        logger.error("No PV mappings available for passive data aggregation")
        return None
    parquet_pattern = os.path.join(passive_pv_path, "**", "*.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    if not parquet_files:
        logger.error("No parquet files found in %s", passive_pv_path)
        return None
    pv_to_location = {}
    for _, row in pv_mappings_df.iterrows():
        pv_id = row["pv_ss_id"]
        location = row["training_location"]
        if location not in pv_to_location:
            pv_to_location[location] = []
        pv_to_location[location].append(pv_id)
    all_aggregated_data = []
    for file_path in tqdm(parquet_files, desc="Loading PV files"):
        try:
            pv_data = pd.read_parquet(file_path)
            if "datetime_GMT" in pv_data.columns:
                pv_data["timestamp"] = pd.to_datetime(pv_data["datetime_GMT"])
            else:
                continue
            all_relevant_pv_ids = []
            for pv_ids in pv_to_location.values():
                all_relevant_pv_ids.extend(pv_ids)
            if "ss_id" in pv_data.columns:
                pv_data_filtered = pv_data[pv_data["ss_id"].isin(all_relevant_pv_ids)]
            else:
                continue
            if pv_data_filtered.empty:
                continue
            pv_data_filtered = pv_data_filtered.copy()
            pv_data_filtered["location"] = pv_data_filtered["ss_id"].map(
                {pv_id: loc for loc, pv_ids in pv_to_location.items() for pv_id in pv_ids}
            )
            if "generation_Wh" in pv_data_filtered.columns:
                pv_data_filtered["generation_mw"] = pv_data_filtered["generation_Wh"] / 1_000_000 / 0.5
                power_col = "generation_mw"
            else:
                continue
            aggregated = pv_data_filtered.groupby(["location", "timestamp"])[power_col].sum().reset_index()
            aggregated.rename(columns={power_col: "passive_pv_generation_mw"}, inplace=True)
            all_aggregated_data.append(aggregated)
        except Exception:
            continue
    if not all_aggregated_data:
        logger.error("No passive PV data could be processed")
        return None
    combined_passive_data = pd.concat(all_aggregated_data, ignore_index=True)
    final_aggregated = combined_passive_data.groupby(["location", "timestamp"])['passive_pv_generation_mw'].sum().reset_index()
    aggregated_path = os.path.join(OUTPUT_DIRECTORY, "aggregated_passive_pv_data_all_locations.parquet")
    final_aggregated.to_parquet(aggregated_path, index=False)
    logger.warning("Saved aggregated data to: %s", aggregated_path)
    return final_aggregated


def merge_passive_pv_with_main_data(main_df, aggregated_passive_df):
    if aggregated_passive_df is None:
        logger.error("No aggregated passive PV data available for merging")
        return main_df
    main_df["timestamp"] = pd.to_datetime(main_df["timestamp"])
    aggregated_passive_df["timestamp"] = pd.to_datetime(aggregated_passive_df["timestamp"])
    merged_df = pd.merge(
        main_df,
        aggregated_passive_df[["location", "timestamp", "passive_pv_generation_mw"]],
        on=["location", "timestamp"],
        how="left",
    )
    merged_df["passive_pv_generation_mw"] = merged_df["passive_pv_generation_mw"].fillna(0)
    return merged_df


if pv_mappings is not None:
    aggregated_passive_data = load_and_aggregate_passive_pv_data(pv_mappings, PASSIVE_PV_DATA_PATH)
else:
    aggregated_passive_data = None


if aggregated_passive_data is not None:
    df_with_passive_pv = merge_passive_pv_with_main_data(df, aggregated_passive_data)
    enhanced_data_path = os.path.join(OUTPUT_DIRECTORY, "tft_ready_data_with_passive_pv_all_locations.parquet")
    df_with_passive_pv.to_parquet(enhanced_data_path, index=False)
    logger.warning("Enhanced dataset saved to: %s", enhanced_data_path)
else:
    df_with_passive_pv = df

