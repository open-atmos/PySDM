# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest


class TestBasicOps:
    @staticmethod
    @pytest.mark.parametrize(
        "output, addend, expected",
        [
            ([1.0], 2, [3.0]),
            ([1.0], [2], [3.0]),
        ],
    )
    def test_addition(backend_instance, output, addend, expected):
        # Arrange
        backend = backend_instance
        output = backend.Storage.from_ndarray(np.asarray(output))
        if hasattr(addend, "__len__"):
            addend = backend.Storage.from_ndarray(np.asarray(addend))

        # Act
        output += addend

        # Assert
        np.testing.assert_array_equal(output.to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize(
        "data",
        (
            [1.0],
            [2.0, 3, 4],
            [-1, np.nan, np.inf],
        ),
    )
    def test_exp(backend_class, data):
        # Arrange
        backend = backend_class(double_precision=True)
        output = backend.Storage.from_ndarray(np.asarray(data))

        # Act
        output.exp()

        # Assert
        np.testing.assert_array_equal(output.to_ndarray(), np.exp(data))

    @staticmethod
    @pytest.mark.parametrize(
        "data, expected",
        [
            ([1, 2], 2),
            ([0, 0], 0),
            ([999, 99, 9], 999),
        ],
    )
    def test_amax(backend_class, data, expected):
        backend = backend_class(double_precision=True)
        output = backend.Storage.from_ndarray(np.asarray(data))

        # Act
        actual = output.amax()

        # Assert
        assert actual == expected
