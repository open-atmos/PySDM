# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.backends import CPU, GPU, GPU_NUMBA


@pytest.fixture(params=(GPU_NUMBA,))
def backend_class(request):
    return request.param
