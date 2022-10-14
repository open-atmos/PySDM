# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from ....backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


class TestArithmetics:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "output, addend, expected",
        [
            ([1.0], 2, [3.0]),
            ([1.0], [2], [3.0]),
        ],
    )
    # pylint: disable=redefined-outer-name
    def test_addition(backend_class, output, addend, expected):
        # Arrange
        backend = backend_class()
        output = backend.Storage.from_ndarray(np.asarray(output))
        if hasattr(addend, "__len__"):
            addend = backend.Storage.from_ndarray(np.asarray(addend))

        # Act
        output += addend

        # Assert
        np.testing.assert_array_equal(output.to_ndarray(), expected)
