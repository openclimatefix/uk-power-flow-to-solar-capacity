"""Tests for src.process.preprocess."""

from __future__ import annotations

import inspect

from process.preprocess import (
    attach_coordinates_to_agg,
    calculate_and_fill_power,
    check_combined_file,
    check_file_variants,
    combine_all_hh_files,
    combine_hh_files_streaming,
    count_unique_tx_ids,
    create_location_aggregated_dataset,
    create_optimized_versions_streaming,
    get_all_unique_tx_ids_alphabetically,
    inspect_geojson_file,
    load_agg_data,
    load_geojson_data,
    main,
    main_power_location_matching,
    match_locations_to_geo,
    quick_file_check,
    rank_all_transformers_by_completeness,
)


def test_preprocess_has_core_functions() -> None:
    expected = [
        combine_all_hh_files,
        combine_hh_files_streaming,
        quick_file_check,
        create_optimized_versions_streaming,
        check_combined_file,
        check_file_variants,
        count_unique_tx_ids,
        get_all_unique_tx_ids_alphabetically,
        rank_all_transformers_by_completeness,
        calculate_and_fill_power,
        create_location_aggregated_dataset,
        inspect_geojson_file,
        load_agg_data,
        load_geojson_data,
        match_locations_to_geo,
        attach_coordinates_to_agg,
        main_power_location_matching,
        main,
    ]
    for fn in expected:
        assert callable(fn)


def test_combine_all_hh_files_signature() -> None:
    sig = inspect.signature(combine_all_hh_files)
    for param in ["cfg", "base_path", "output_path", "sample_run"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_combine_hh_files_streaming_signature() -> None:
    sig = inspect.signature(combine_hh_files_streaming)
    for param in ["cfg", "base_path", "output_path", "chunk_size"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_calculate_and_fill_power_signature() -> None:
    sig = inspect.signature(calculate_and_fill_power)
    for param in ["cfg", "input_filepath", "output_filepath", "transformer_id", "power_factor"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_match_locations_to_geo_signature() -> None:
    sig = inspect.signature(match_locations_to_geo)
    for param in ["cfg", "df_agg", "df_geo", "fuzzy_threshold"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_all_functions_accept_cfg_as_first_param() -> None:
    cfg_fns = [
        combine_all_hh_files,
        combine_hh_files_streaming,
        quick_file_check,
        create_optimized_versions_streaming,
        check_combined_file,
        check_file_variants,
        count_unique_tx_ids,
        get_all_unique_tx_ids_alphabetically,
        rank_all_transformers_by_completeness,
        calculate_and_fill_power,
        create_location_aggregated_dataset,
        inspect_geojson_file,
        load_agg_data,
        load_geojson_data,
        match_locations_to_geo,
        main_power_location_matching,
    ]
    for fn in cfg_fns:
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        assert params[0] == "cfg", f"{fn.__name__} first param should be cfg, got {params[0]}"
