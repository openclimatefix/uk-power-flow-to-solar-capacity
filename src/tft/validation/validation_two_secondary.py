import pandas as pd
import os
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENHANCED_DATA_PATH = "/home/felix/output/tft_ready_data_with_passive_pv_all_locations.parquet"
OUTPUT_DIRECTORY = "/home/felix/output/"

logger.warning("Creating final dataset for all locations")
logger.warning("Loading enhanced dataset from: %s", ENHANCED_DATA_PATH)

df_enhanced = pd.read_parquet(ENHANCED_DATA_PATH)
logger.warning("Loaded enhanced dataset. Shape: %s", str(df_enhanced.shape))

all_locations = df_enhanced["location"].unique()
logger.warning("Found %d unique locations in dataset", len(all_locations))
for i, location in enumerate(sorted(all_locations), 1):
    location_count = len(df_enhanced[df_enhanced["location"] == location])
    logger.info("%2d. %s: %s rows", i, location, f"{location_count:,}")

logger.warning("Checking passive PV data availability across all locations")
df_enhanced["timestamp"] = pd.to_datetime(df_enhanced["timestamp"])

locations_with_pv = []
locations_without_pv = []

for location in all_locations:
    location_data = df_enhanced[df_enhanced["location"] == location]
    non_zero_pv = location_data[location_data["passive_pv_generation_mw"] > 0]
    if len(non_zero_pv) > 0:
        locations_with_pv.append(location)
        pv_start = non_zero_pv["timestamp"].min()
        pv_end = non_zero_pv["timestamp"].max()
        logger.info("%s: PV data from %s to %s (%s non-zero records)", location, pv_start, pv_end, f"{len(non_zero_pv):,}")
    else:
        locations_without_pv.append(location)

tmin, tmax = df_enhanced["timestamp"].min(), df_enhanced["timestamp"].max()
logger.warning("Input window: %s → %s", tmin, tmax)

logger.warning("PV Data Summary: with=%d/%d (%.1f%%), without=%d", len(locations_with_pv), len(all_locations), 100 * len(locations_with_pv) / len(all_locations), len(locations_without_pv))
if locations_without_pv:
    for loc in sorted(locations_without_pv):
        logger.info("No passive PV data for: %s", loc)

logger.warning("Overall passive PV data coverage")
total_records = len(df_enhanced)
records_with_pv_column = df_enhanced["passive_pv_generation_mw"].notna().sum()
records_with_nonzero_pv = (df_enhanced["passive_pv_generation_mw"] > 0).sum()
logger.warning("Total records: %s", f"{total_records:,}")
logger.warning("Records with PV column: %s (%.1f%%)", f"{records_with_pv_column:,}", 100 * records_with_pv_column / total_records)
logger.warning("Records with non-zero PV: %s (%.2f%%)", f"{records_with_nonzero_pv:,}", 100 * records_with_nonzero_pv / total_records)

df_final = df_enhanced.copy()
logger.warning("Using complete dataset with all locations. Shape: %s", str(df_final.shape))

original_columns = [col for col in df_final.columns if col != "passive_pv_generation_mw"]
logger.info("Final dataset columns (%d total)", len(df_final.columns))
logger.info("Original columns: %d", len(original_columns))
logger.info("New column: passive_pv_generation_mw")

logger.warning("Computing net power columns with scaling factors for all locations")

location_pv_peaks = df_final.groupby("location")["passive_pv_generation_mw"].max()
logger.info("Top 10 locations by peak PV generation:\n%s", location_pv_peaks.sort_values(ascending=False).head(10).to_string())

logger.warning("Adding net power columns for all locations")

df_final["v2"] = 0.0
df_final["v2_1p0"] = 0.0
df_final["v2_0p5"] = 0.0
df_final["v2_0p1"] = 0.0

locations_processed = 0
locations_with_scaling = 0

for location in all_locations:
    location_mask = df_final["location"] == location
    location_data = df_final[location_mask]
    if len(location_data) == 0:
        continue
    locations_processed += 1
    current_peak_pv = location_data["passive_pv_generation_mw"].max()
    df_final.loc[location_mask, "v2"] = (
        df_final.loc[location_mask, "active_power_mw"] - df_final.loc[location_mask, "passive_pv_generation_mw"]
    )
    if current_peak_pv > 0:
        locations_with_scaling += 1
        scale_factor_1p0 = 1.0 / current_peak_pv
        scaled_pv_1p0 = df_final.loc[location_mask, "passive_pv_generation_mw"] * scale_factor_1p0
        df_final.loc[location_mask, "v2_1p0"] = df_final.loc[location_mask, "active_power_mw"] - scaled_pv_1p0
        scale_factor_0p5 = 0.5 / current_peak_pv
        scaled_pv_0p5 = df_final.loc[location_mask, "passive_pv_generation_mw"] * scale_factor_0p5
        df_final.loc[location_mask, "v2_0p5"] = df_final.loc[location_mask, "active_power_mw"] - scaled_pv_0p5
        scale_factor_0p1 = 0.1 / current_peak_pv
        scaled_pv_0p1 = df_final.loc[location_mask, "passive_pv_generation_mw"] * scale_factor_0p1
        df_final.loc[location_mask, "v2_0p1"] = df_final.loc[location_mask, "active_power_mw"] - scaled_pv_0p1
    else:
        df_final.loc[location_mask, "v2_1p0"] = df_final.loc[location_mask, "active_power_mw"]
        df_final.loc[location_mask, "v2_0p5"] = df_final.loc[location_mask, "active_power_mw"]
        df_final.loc[location_mask, "v2_0p1"] = df_final.loc[location_mask, "active_power_mw"]

logger.warning("Processed %d locations; with PV scaling: %d; using active_power_mw only: %d", locations_processed, locations_with_scaling, locations_processed - locations_with_scaling)

logger.warning("Statistics for new net power columns (all locations)")
new_columns = ["v2", "v2_1p0", "v2_0p5", "v2_0p1"]
for col in new_columns:
    stats = df_final[col]
    logger.info("%s: mean=%.4f min=%.4f max=%.4f std=%.4f MW", col, stats.mean(), stats.min(), stats.max(), stats.std())

logger.warning("Final dataset shape: %s", str(df_final.shape))

sample_cols = [
    "location",
    "timestamp",
    "active_power_mw",
    "passive_pv_generation_mw",
    "v2",
    "v2_1p0",
    "v2_0p5",
    "v2_0p1",
]

if all(col in df_final.columns for col in sample_cols):
    top_pv_locations_list = location_pv_peaks.sort_values(ascending=False).head(15).index.tolist()
    sample_data = df_final[df_final["location"].isin(top_pv_locations_list)].head(25)
    logger.info("Sample rows from high-PV locations:\n%s", sample_data[sample_cols].to_string(index=False))
else:
    available_cols = [col for col in sample_cols if col in df_final.columns]
    logger.info("Sample columns not all available; showing first 5 rows of available columns:\n%s", df_final[available_cols].head().to_string(index=False))


tmin_final, tmax_final = df_final["timestamp"].min(), df_final["timestamp"].max()
date_tag = tmax_final.strftime("%Y-%m-%d")
final_output_path = os.path.join(OUTPUT_DIRECTORY, f"all_locations_tft_ready_with_passive_pv_{date_tag}.parquet")

_tmp = final_output_path + ".tmp"
df_final.to_parquet(_tmp, index=False)
os.replace(_tmp, final_output_path)

latest_output_path = os.path.join(OUTPUT_DIRECTORY, "all_locations_tft_ready_with_passive_pv.parquet")
_tmp_latest = latest_output_path + ".tmp"
df_final.to_parquet(_tmp_latest, index=False)
os.replace(_tmp_latest, latest_output_path)

logger.warning("Final dataset saved: %s", final_output_path)
logger.warning("Also updated: %s", latest_output_path)

df_check = pd.read_parquet(latest_output_path, columns=["timestamp", "location", "v2", "v2_1p0", "v2_0p5", "v2_0p1", "passive_pv_generation_mw"])
logger.warning("Reloaded final: rows=%s window=%s → %s locs=%d", f"{len(df_check):,}", df_check["timestamp"].min(), df_check["timestamp"].max(), df_check["location"].nunique())
missing = [c for c in ["v2", "v2_1p0", "v2_0p5", "v2_0p1"] if c not in df_check.columns]
if missing:
    raise AssertionError(f"Missing expected columns after save: {missing}")

location_summary = (
    df_final.groupby("location").agg({
        "timestamp": "count",
        "passive_pv_generation_mw": ["max", "mean"],
        "v2": "mean",
    }).round(4)
)
location_summary.columns = ["record_count", "max_pv_mw", "mean_pv_mw", "mean_net_power_v2"]
location_summary = location_summary.sort_values("max_pv_mw", ascending=False)
logger.info("Top 15 locations by max PV generation:\n%s", location_summary.head(15).to_string())

logger.warning("All locations dataset creation completed")
logger.warning("Final Summary: total_locations=%d total_records=%s with_pv=%d new_columns=%d output=%s", len(all_locations), f"{len(df_final):,}", len(locations_with_pv), len(new_columns), final_output_path)
