# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import Numba, ThrustRTC


@pytest.mark.parametrize(
    "backend", (pytest.param(ThrustRTC, marks=pytest.mark.xfail(strict=True)), Numba)
)
def test_setitem(backend):
    # arrange
    arr = backend.Storage.from_ndarray(np.zeros(3))

    # act
    arr[1] = 1

    # assert
    assert arr[1] == 1
    assert arr[0] == arr[2] == 0
