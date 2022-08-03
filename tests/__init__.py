import gc
import platform

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    if platform.architecture()[0] != "32bit":
        gc.collect()
