from typing import Sequence

import numpy as np
import pytest

from PySDM.storages.numba.storage import Storage


@pytest.mark.parametrize(
    "output, addend, expected",
    [
        ([1.0], 2, [3.0]),
        ([1.0], [2], [3.0]),
    ],
)
def test_addition(output, addend, expected):
    # Arrange
    output = Storage.from_ndarray(np.asarray(output))
    if isinstance(addend, Sequence):
        addend = Storage.from_ndarray(np.asarray(addend))

    # Act
    output += addend

    # Assert
    np.testing.assert_array_equal(output.to_ndarray(), expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [1.0]),
        ([-3.0], 3.0, [-27.0]),
        ([-0.5], -0.75, [-(0.5**-0.75)]),
    ],
)
def test_power(data, power, expected):
    # Arrange
    storage = Storage.from_ndarray(np.asarray(data))

    # Act
    storage **= power

    # Assert
    np.testing.assert_array_equal(storage.data, expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [2.0]),
        ([-3.0], 3.0, [-9.0]),
        ([-0.5], -0.75, [-0.5 * -0.75]),
    ],
)
def test_multiply(data, power, expected):
    # Arrange
    storage = Storage.from_ndarray(np.asarray(data))

    # Act
    storage *= power

    # Assert
    np.testing.assert_array_equal(storage.data, expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [0.5]),
        ([-3.0], 3.0, [-1.0]),
        ([-0.5], -0.75, [-0.5 / -0.75]),
    ],
)
def test_divide(data, power, expected):
    # Arrange
    storage = Storage.from_ndarray(np.asarray(data))

    # Act
    storage /= power

    # Assert
    np.testing.assert_array_equal(storage.data, expected)


def test_row_modulo():
    # Arrange
    data = np.random.randint(8, size=(2, 4), dtype=np.int32)
    modulo = np.random.randint(1, 9, size=(2, 1), dtype=np.int32)
    storage = Storage.from_ndarray(data)
    second_storage = Storage.from_ndarray(modulo)

    # Act
    storage %= second_storage

    # Assert
    np.testing.assert_array_almost_equal(storage.data, data % modulo)


@pytest.mark.parametrize(
    "data, subtrahend, expected",
    [
        ([1.0], [2.0], [-1.0]),
        ([-3.0], [3.0], [-6.0]),
        ([-0.5, 0.75], [-0.75, 2.5], [0.25, -1.75]),
    ],
)
def test_subtraction(data, subtrahend, expected):
    # Arrange
    storage = Storage.from_ndarray(np.asarray(data))

    # Act
    storage -= Storage.from_ndarray(np.asarray(subtrahend))

    # Assert
    np.testing.assert_array_equal(storage.data, expected)
