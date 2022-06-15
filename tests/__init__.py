import gc

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()
