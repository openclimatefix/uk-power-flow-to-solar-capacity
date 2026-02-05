import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process.sanity as sanity  # noqa: E402


def test_sanity_has_main_and_run_sanity_checks():
    assert hasattr(sanity, "run_sanity_checks")
    assert callable(sanity.run_sanity_checks)
    assert hasattr(sanity, "main")
    assert callable(sanity.main)

