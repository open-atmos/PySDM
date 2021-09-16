import numpy as np
import pytest
from PySDM.backends import CPU, GPU


@pytest.mark.parametrize("backend", (
    pytest.param(GPU, marks=pytest.mark.xfail(strict=True)),
    CPU
))
def test_setitem(backend):
    # arrange
    arr = backend.Storage.from_ndarray(np.zeros(3))

    # act
    arr[1] = 1

    # assert
    assert arr[1] == 1
    assert arr[0] == arr[2] == 0
