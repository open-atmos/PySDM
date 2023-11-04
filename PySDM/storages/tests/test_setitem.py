import numpy as np
import pytest


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_setitem(storage_class):
    # Arrange
    arr = storage_class.from_ndarray(np.zeros(3))

    # Act
    arr[1] = 1

    # Assert
    assert arr[1] == 1
    assert arr[0] == arr[2] == 0
