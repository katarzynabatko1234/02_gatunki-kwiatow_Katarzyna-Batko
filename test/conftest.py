import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C  # noqa: E402

HAS_DATA = C.CACHE_DIR.exists() and C.SPLIT_CSV.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="brak danych: uruchom najpierw `python src/data.py`")


@pytest.fixture(scope="session")
def df():
    if not HAS_DATA:
        pytest.skip("brak danych")
    from data import prepare
    return prepare()
