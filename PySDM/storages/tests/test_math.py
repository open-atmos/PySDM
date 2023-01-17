from typing import Sequence

import numpy as np
import pytest


@pytest.mark.parametrize(
    "output, addend, expected",
    [
        ([1.0], 2, [3.0]),
        ([1.0], [2], [3.0]),
    ],
)
def test_addition(storage_class, output, addend, expected):
    # Arrange
    output = storage_class.from_ndarray(np.asarray(output))
    if isinstance(addend, Sequence):
        addend = storage_class.from_ndarray(np.asarray(addend))

    # Act
    output += addend

    # Assert
    np.testing.assert_array_equal(output.to_ndarray(), expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [1.0]),
        ([-3.0], 3.0, [-27.0]),
        ([0.5], -0.75, [0.5**-0.75]),
    ],
)
def test_power(storage_class, data, power, expected):
    # Arrange
    storage = storage_class.from_ndarray(np.asarray(data))

    # Act
    storage **= power

    # Assert
    np.testing.assert_allclose(storage.to_ndarray(), expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [2.0]),
        ([-3.0], 3.0, [-9.0]),
        ([-0.5], -0.75, [-0.5 * -0.75]),
    ],
)
def test_multiply(storage_class, data, power, expected):
    # Arrange
    storage = storage_class.from_ndarray(np.asarray(data))

    # Act
    storage *= power

    # Assert
    np.testing.assert_array_equal(storage.to_ndarray(), expected)


@pytest.mark.parametrize(
    "data, power, expected",
    [
        ([1.0], 2.0, [0.5]),
        ([-3.0], 3.0, [-1.0]),
        ([-0.5], -0.75, [-0.5 / -0.75]),
    ],
)
def test_divide(storage_class, data, power, expected):
    # Arrange
    storage = storage_class.from_ndarray(np.asarray(data))

    # Act
    storage /= power

    # Assert
    np.testing.assert_allclose(storage.to_ndarray(), expected)


@pytest.mark.parametrize(
    "data, subtrahend, expected",
    [
        ([1.0], [2.0], [-1.0]),
        ([-3.0], [3.0], [-6.0]),
        ([-0.5, 0.75], [-0.75, 2.5], [0.25, -1.75]),
    ],
)
def test_subtraction(storage_class, data, subtrahend, expected):
    # Arrange
    storage = storage_class.from_ndarray(np.asarray(data))

    # Act
    storage -= storage_class.from_ndarray(np.asarray(subtrahend))

    # Assert
    np.testing.assert_allclose(storage.to_ndarray(), expected)


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_row_modulo(storage_class):
    # Arrange
    data = np.random.randint(8, size=(2, 4), dtype=np.int32)
    modulo = np.random.randint(1, 9, size=2, dtype=np.int32)
    storage = storage_class.from_ndarray(data)
    second_storage = storage_class.from_ndarray(modulo)

    # Act
    storage %= second_storage

    # Assert
    np.testing.assert_array_equal(storage.data, (data.T % modulo).T)
