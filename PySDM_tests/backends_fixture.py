import pytest
from PySDM.backends import CPU, GPU


@pytest.fixture(params=[
    CPU,
    GPU
])
def backend(request):
    return request.param
