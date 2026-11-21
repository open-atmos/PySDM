"""skips tests parametrised with the GPU (ThrustRTC) backend if CUDA is not available"""

import sys

import pytest


def pytest_collection_modifyitems(items):
    # PySDM is not imported here, as some tests (e.g., devops_tests) run without its
    # dependencies; if no collected test imported PySDM.backends, none uses the GPU backend
    backends = sys.modules.get("PySDM.backends")
    if backends is None or backends.ThrustRTC.ENABLE:
        return

    def uses_gpu(value):
        return (
            value is backends.GPU
            or value is backends.ThrustRTC
            or (isinstance(value, str) and value == "GPU")
        )

    skip_gpu = pytest.mark.skip(reason="CUDA not available (GPU backend disabled)")
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and any(
            uses_gpu(value) for value in callspec.params.values()
        ):
            item.add_marker(skip_gpu)
