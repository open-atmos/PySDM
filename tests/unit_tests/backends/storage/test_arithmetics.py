# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest


class TestArithmetics:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1.0], 2, [3.0]),
            ([1.0], [2], [3.0]),
        ],
    )
    def test_addition(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a += b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)

        return

    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1.0], 2, [-1.0]),
            ([1.0], [2], [-1.0]),
        ],
    )
    def test_subtract(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a -= b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1.0], 2, [2.0]),
            ([1.0], [2], [2.0]),
        ],
    )
    def test_multiply(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a *= b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([3], 2, [1]),
            ([3.0], [2], [1.5]),
        ],
    )
    def test_truediv(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a /= b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([5], 2, [1]),
            ([5], [2], [1]),
        ],
    )
    def test_modulo(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a %= b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([5], 2, [25]),
            ([4.0], [1.5], [8.0]),
        ],
    )
    def test_pow(backend_class, a, b, expected):
        # Arrange
        backend = backend_class()
        a = backend.Storage.from_ndarray(np.asarray(a))
        if hasattr(b, "__len__"):
            b = backend.Storage.from_ndarray(np.asarray(b))

        # Act
        a **= b

        # Assert
        np.testing.assert_array_equal(a.to_ndarray(), expected)
