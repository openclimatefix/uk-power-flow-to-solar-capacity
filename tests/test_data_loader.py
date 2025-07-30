import os
import zipfile

import pandas as pd

from src.data_loader import load_csv_data, load_era5_data, unzip_era5_files


def test_unzip_era5_files(tmp_path):
    """
    Tests that the unzipping function correctly extracts .nc files.
    """
    zip_dir = tmp_path / "zip_folder"
    extract_dir = tmp_path / "extract_folder"
    zip_dir.mkdir()
    extract_dir.mkdir()

    zip_filename = "test_archive.zip"
    zip_path = zip_dir / zip_filename

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy_file.nc", b"netcdf_content")
        zf.writestr("other_file.txt", b"text_content")

    unzip_era5_files(str(zip_dir), [zip_filename], str(extract_dir))

    # Check that only .nc file was extracted
    extracted_files = os.listdir(extract_dir)
    assert len(extracted_files) == 1
    assert extracted_files[0].startswith("era5_data_0_")
    assert extracted_files[0].endswith(".nc")


def test_load_csv_data_success(test_config):
    """Tests the successful loading and basic processing of CSV files."""
    paths = test_config["paths"]
    power_cols = test_config["data_ingestion_params"]["power_csv_cols"]

    df_power, df_sites = load_csv_data(paths["power_flow_path"], paths["sites_path"], power_cols)

    # Assertions - power data
    assert df_power is not None
    assert not df_power.empty
    assert set(df_power.columns) == {"timestamp", "power", "tx_id"}
    assert pd.api.types.is_datetime64_any_dtype(df_power["timestamp"])
    assert pd.api.types.is_numeric_dtype(df_power["power"])
    assert df_power["power"].iloc[0] == 3.5

    # Assertions - sites data
    assert df_sites is not None
    assert not df_sites.empty
    assert "SiteName" in df_sites.columns


def test_load_csv_data_file_not_found(test_config):
    """Tests that the function handles a missing file gracefully."""
    paths = test_config["paths"]
    power_cols = test_config["data_ingestion_params"]["power_csv_cols"]

    df_power, df_sites = load_csv_data("non_existent_file.csv", paths["sites_path"], power_cols)

    # Assert - return None for both dataframes if one absent
    assert df_power is None
    assert df_sites is None


def test_load_era5_data_success(test_config):
    """Tests the successful loading of ERA5 NetCDF data."""
    paths = test_config["paths"]

    ds_era5 = load_era5_data(paths["era5_extract_dir"], paths["skt_files_path"])

    assert ds_era5 is not None
    assert "t2m" in ds_era5.data_vars
    assert "latitude" in ds_era5.coords
    assert "longitude" in ds_era5.coords
    assert ds_era5["time"].size > 0
