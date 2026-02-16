import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process.preprocess as preprocess  # noqa: E402


def test_preprocess_imports():
    assert hasattr(preprocess, "combine_hh_files_streaming")


def test_preprocess_has_core_functions():
    expected_funcs = [
        "combine_all_hh_files",
        "combine_hh_files_streaming",
        "quick_file_check",
        "create_optimized_versions_streaming",
        "check_combined_file",
        "check_file_variants",
        "count_unique_tx_ids",
        "get_all_unique_tx_ids_alphabetically",
        "rank_all_transformers_by_completeness",
        "calculate_and_fill_power",
        "create_location_aggregated_dataset",
        "inspect_geojson_file",
        "load_agg_data",
        "load_geojson_data",
        "match_locations_to_geo",
        "attach_coordinates_to_agg",
        "main_power_location_matching",
        "main",
    ]
    for name in expected_funcs:
        assert hasattr(preprocess, name), f"Missing function: {name}"


def test_config_objects_present():
    for name in ["CFG", "PATHS", "PROCESSING", "MATCHING"]:
        assert hasattr(preprocess, name), f"Missing config object: {name}"


def test_paths_have_expected_attributes():
    paths = preprocess.PATHS
    for attr in [
        "base_data_dir",
        "zip_filename",
        "extract_dirname",
        "agg_csv",
        "geojson",
        "out_match",
        "out_unmatched",
        "combined_full_csv",
        "combined_power_only_csv",
        "combined_reduced_csv",
        "combined_filled_power_csv",
        "combined_aggregated_location_csv",
        "aggregated_with_coords_csv",
    ]:
        assert hasattr(paths, attr), f"PATHS missing attribute: {attr}"


def test_processing_has_expected_attributes():
    processing = preprocess.PROCESSING
    for attr in [
        "fuzzy_threshold",
        "sample_run",
        "sample_file_limit",
        "chunk_size_streaming",
        "chunk_size_quick_check",
        "chunk_size_optimized",
        "chunk_size_count_unique",
        "chunk_size_rank",
        "chunk_size_fill_power",
        "chunk_size_aggregate_location",
        "power_factor",
    ]:
        assert hasattr(processing, attr), f"PROCESSING missing attribute: {attr}"


def test_fuzzy_threshold_is_float():
    assert isinstance(preprocess.FUZZY_THRESHOLD, float)


def test_combine_all_hh_files_signature_stable():
    sig = inspect.signature(preprocess.combine_all_hh_files)
    for param in ["base_path", "output_path", "sample_run"]:
        assert param in sig.parameters


def test_combine_hh_files_streaming_signature_stable():
    sig = inspect.signature(preprocess.combine_hh_files_streaming)
    for param in ["base_path", "output_path", "chunk_size"]:
        assert param in sig.parameters
