import numpy as np

from PySDM.storages.numba.storage import Storage


def test_setitem():
    # Arrange
    arr = Storage.from_ndarray(np.zeros(3))

    # Act
    arr[1] = 1

    # Assert
    assert arr[1] == 1
    assert arr[0] == arr[2] == 0
