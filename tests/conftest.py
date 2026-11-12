"""skips tests parametrised with the GPU (ThrustRTC) backend if CUDA is not available"""

import pytest

from PySDM.backends import GPU, ThrustRTC


def _uses_gpu(value):
    return (
        value is GPU
        or value is ThrustRTC
        or (isinstance(value, str) and value == "GPU")
    )


def pytest_collection_modifyitems(items):
    if ThrustRTC.ENABLE:
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available (GPU backend disabled)")
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and any(
            _uses_gpu(value) for value in callspec.params.values()
        ):
            item.add_marker(skip_gpu)
