"""Functionality for cleaning and transforming raw data."""

import logging
import re

import pandas as pd
import pyproj
import xarray as xr


def get_site_coordinates(
    tx_id: str, df_sites: pd.DataFrame,
) -> dict[str, float] | None:
    """Finds and converts site coordinates from Easting/Northing to Latitude/Longitude."""
    if df_sites is None:
        return None

    # Match patterns such as 'BUSH_HOLLOW_11kv_t1'
    match = re.match(
        r"^(.*?)_?(?:primary_)?(?:local_)?(?:grid_)?\d+(\.\d+)?kv_t\d+[a-zA-Z]*",
        tx_id,
        re.IGNORECASE,
    )
    if not match:
        # Fallback for simpler names  i.e. 'SITE_A_11kv'
        match = re.match(r"^(.*?)_?\d+(\.\d+)?kv", tx_id, re.IGNORECASE)

    search_name = match.group(1).replace("_", " ").strip() if match else tx_id.split("_")[0]
    regex_pattern = r"(?i)\b" + re.escape(search_name).replace("-", r"[- ]") + r"\b"

    potential_rows = df_sites[
        df_sites["SiteName"].astype(str).str.contains(regex_pattern, regex=True, na=False)
    ]

    if potential_rows.empty:
        logging.warning("No coordinates found for site matching name: %s", search_name)
        return None

    row = potential_rows.iloc[0]
    if pd.notna(row["Easting"]) and pd.notna(row["Northing"]):
        transformer = pyproj.Transformer.from_crs(
            "EPSG:27700", "EPSG:4326", always_xy=True,
        )
        lon, lat = transformer.transform(row["Easting"], row["Northing"])
        logging.info("Coordinates found for %s: Lat %.4f, Lon %.4f", tx_id, lat, lon)
        return {"latitude": lat, "longitude": lon}

    return None


def process_single_site_power(
    tx_id: str, df_power_full: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp,
) -> pd.Series | None:
    """Filters, cleans, and resamples power data for a single transformer site."""
    df_site = df_power_full[df_power_full["tx_id"] == tx_id].copy()
    if df_site.empty:
        logging.warning("No raw power data found for %s.", tx_id)
        return None

    df_site.set_index("timestamp", inplace=True)
    if df_site.index.tz is None:
        df_site = df_site.tz_localize("UTC", ambiguous="infer")
    else:
        df_site = df_site.tz_convert("UTC")

    df_site = df_site[~df_site.index.duplicated(keep="first")]
    power_series = df_site["power"].resample("h").mean().abs()

    filtered_series = power_series[
        (power_series.index >= start_dt) & (power_series.index <= end_dt)
    ]

    if filtered_series.empty:
        logging.warning("Power data for %s is empty after date filtering.", tx_id)
        return None

    logging.info(
        "Processed active power for %s (%d hourly records).",
        tx_id,
        len(filtered_series),
    )
    return filtered_series


def get_site_era5_data(
    coords: dict[str, float] | None,
    ds_era5: xr.Dataset | None,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame | None:
    """Selects ERA5 weather data for a specific site's coordinates and date range."""
    if ds_era5 is None or coords is None:
        return None

    time_coord = next(
        (name for name in ["time", "valid_time"] if name in ds_era5.coords), None,
    )
    lat_coord = "latitude" if "latitude" in ds_era5.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds_era5.coords else "lon"

    era5_start_naive = start_dt.tz_localize(None)
    era5_end_naive = end_dt.tz_localize(None)

    ds_site = (
        ds_era5.sel(
            {lat_coord: coords["latitude"], lon_coord: coords["longitude"]},
            method="nearest",
        )
        .sel({time_coord: slice(era5_start_naive, era5_end_naive)})
        .load()
    )

    if ds_site[time_coord].size == 0:
        logging.warning("No ERA5 data found for site after time/location selection.")
        return None

    logging.info(
        "Loaded ERA5 data for coordinates (timeseries length: %d).",
        ds_site[time_coord].size,
    )
    return ds_site.to_dataframe()


def _interpolate_power(group: pd.DataFrame) -> pd.DataFrame:
    """Helper function to interpolate the power column for a site group."""
    if "power" in group.columns and group["power"].isnull().any():
        group["power"] = (
            group["power"]
            .interpolate(method="linear", limit_direction="both", limit=24)
            .ffill()
            .bfill()
        )
    return group


def _interpolate_weather(group: pd.DataFrame, era5_vars: list[str]) -> pd.DataFrame:
    """Helper function to interpolate weather columns for a site group."""
    tx_id = group["tx_id"].iloc[0]
    for var in era5_vars:
        if var not in group.columns or not group[var].isnull().any():
            continue

        if group[var].isnull().all():
            if var == "skt" and "t2m" in group.columns and group["t2m"].notnull().any():
                logging.warning(
                    "Variable '%s' for site %s is all NaNs. Filling with 't2m'.",
                    var,
                    tx_id,
                )
                group[var] = group["t2m"]
            else:
                logging.error(
                    "Variable '%s' for site %s is all NaNs. No proxy. Filling with 0.",
                    var,
                    tx_id,
                )
                group[var] = group[var].fillna(0)
        else:
            group[var] = (
                group[var]
                .interpolate(method="linear", limit_direction="both", limit=24)
                .ffill()
                .bfill()
            )
    return group


def handle_missing_values(df: pd.DataFrame, era5_vars: list[str]) -> pd.DataFrame:
    """Interpolates missing values in the master DataFrame on a per-site basis."""

    def interpolate_group(group: pd.DataFrame) -> pd.DataFrame:
        group = _interpolate_power(group)
        group = _interpolate_weather(group, era5_vars)
        return group

    df_interpolated = df.groupby("tx_id", group_keys=False).apply(interpolate_group)

    final_nans = df_interpolated.isnull().sum()
    for col, nan_count in final_nans[final_nans > 0].items():
        logging.error(
            "%d NaNs remain in column '%s' after all interpolation steps.",
            nan_count,
            col,
        )

    return df_interpolated


def create_master_dataframe(
    target_ids: list[str],
    df_power: pd.DataFrame,
    df_sites: pd.DataFrame,
    ds_era5: xr.Dataset,
    start_date: str,
    end_date: str,
    era5_vars: list[str],
) -> pd.DataFrame:
    """Orchestrates the preprocessing pipeline to create a single, clean master DataFrame."""
    all_site_dfs = []
    start_dt_utc = pd.to_datetime(start_date, utc=True)
    end_dt_utc = pd.to_datetime(end_date, utc=True).replace(
        hour=23, minute=59, second=59,
    )

    for tx_id in target_ids:
        logging.info("Processing site: %s", tx_id)

        power_series = process_single_site_power(
            tx_id, df_power, start_dt_utc, end_dt_utc,
        )
        if power_series is None:
            logging.warning("Skipping site %s due to lack of power data.", tx_id)
            continue

        coords = get_site_coordinates(tx_id, df_sites)
        df_era5_site = get_site_era5_data(coords, ds_era5, start_dt_utc, end_dt_utc)

        df_site_master = pd.DataFrame(power_series).rename(columns={0: "power"})
        df_site_master["tx_id"] = tx_id

        if df_era5_site is not None:
            df_site_master = pd.merge(
                df_site_master.reset_index(),
                df_era5_site.reset_index(),
                on="timestamp",
                how="left",
            ).set_index("timestamp")

        all_site_dfs.append(df_site_master)

    if not all_site_dfs:
        logging.error("No data could be processed. Master DataFrame is empty.")
        return pd.DataFrame()

    master_df = pd.concat(all_site_dfs).reset_index()
    logging.info(
        "Master DataFrame created with %d records for %d sites before NaN handling.",
        len(master_df),
        len(all_site_dfs),
    )
    master_df_clean = handle_missing_values(master_df, era5_vars)

    return master_df_clean.set_index("timestamp").sort_index()
