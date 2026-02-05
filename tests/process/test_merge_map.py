import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process.merge_map as merge_map  # noqa: E402


def test_merge_map_has_config_objects():
    assert hasattr(merge_map, "CFG")
    assert hasattr(merge_map, "PATHS")
    assert hasattr(merge_map.CFG, "weather")
    assert hasattr(merge_map.CFG.weather, "time_coord")


def test_create_combined_power_weather_parquet_is_callable():
    assert hasattr(merge_map, "create_combined_power_weather_parquet")
    fn = merge_map.create_combined_power_weather_parquet
    assert callable(fn)
