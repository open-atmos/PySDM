# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.backends import CPU, GPU


@pytest.fixture(params=(CPU, GPU))
def backend_class(request):
    return request.param


@pytest.fixture(
    params=(pytest.param(CPU(), id="CPU"), pytest.param(GPU(), id="GPU")),
    scope="session",
)
def backend_instance(request):
    return request.param
