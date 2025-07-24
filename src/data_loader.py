# Functionality for loading raw data from disk

import glob
import logging
import os
import zipfile

import pandas as pd
import xarray as xr


def unzip_era5_files(zip_folder, zip_filenames, extract_dir):
    """
    Extracts .nc files from a list of zip archives if not already present.
    Each extracted file is given a unique name to prevent collisions.
    """
    os.makedirs(extract_dir, exist_ok=True)

    if len(glob.glob(os.path.join(extract_dir, '*.nc'))) >= len(zip_filenames):
        logging.info("ERA5 .nc files appear to be already extracted in %s.", extract_dir)
        return

    logging.info("Extraction required. Processing zip files...")
    for zip_idx, zip_filename in enumerate(zip_filenames):
        zip_file_path = os.path.join(zip_folder, zip_filename)
        if not os.path.exists(zip_file_path):
            logging.warning("Zip file not found: '%s'. Skipping.", zip_file_path)
            continue

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member_info in zip_ref.infolist():
                if member_info.filename.endswith('.nc') and not member_info.is_dir():
                    new_filename = f"era5_data_{zip_idx}_{os.path.basename(member_info.filename)}"
                    target_path = os.path.join(extract_dir, new_filename)
                    if not os.path.exists(target_path):
                        # Extract directly to the final destination
                        with zip_ref.open(member_info) as source, open(target_path, 'wb') as target:
                            target.write(source.read())

    logging.info("Extraction process check completed.")


def load_era5_data(extract_dir, skt_path_pattern):
    """
    Loads all .nc files from the extraction directory and integrates 'skt' files.
    """
    nc_files = glob.glob(os.path.join(extract_dir, '*.nc'))
    skt_nc_files = glob.glob(skt_path_pattern)
    all_nc_files = sorted(list(set(nc_files + skt_nc_files)))

    if not all_nc_files:
        logging.error("No NetCDF files found for ERA5 data. Data will be unavailable.")
        return None

    if skt_nc_files:
         logging.info("Found %d 'skt' NetCDF files to integrate.", len(skt_nc_files))

    def drop_expver(ds):
        return ds.drop_vars('expver', errors='ignore') if 'expver' in ds.coords else ds

    ds_era5 = xr.open_mfdataset(
        all_nc_files,
        combine='by_coords',
        preprocess=drop_expver,
        engine='netcdf4',
        chunks={'time': 'auto'}
    )
    logging.info("ERA5 data loaded with variables: %s", list(ds_era5.data_vars))
    return ds_era5


def load_csv_data(power_path, sites_path, power_cols, sites_encoding='latin1'):
    """
    Loads the transformer power flow and site coordinates CSV files.
    """
    if not os.path.exists(power_path) or not os.path.exists(sites_path):
        logging.error("Power flow or sites coordinate file not found.")
        return None, None

    # Load power data
    df_power = pd.read_csv(power_path, usecols=power_cols.keys())
    df_power.rename(columns=power_cols, inplace=True)
    df_power['timestamp'] = pd.to_datetime(df_power['timestamp'], errors='coerce')
    df_power['power'] = pd.to_numeric(df_power['power'], errors='coerce')
    df_power.dropna(subset=['timestamp', 'power'], inplace=True)
    logging.info("Power flow CSV loaded and preprocessed from %s.", power_path)

    # Load sites data
    df_sites = pd.read_csv(sites_path, encoding=sites_encoding)
    logging.info("Sites coordinate CSV loaded from %s.", sites_path)

    return df_power, df_sites
