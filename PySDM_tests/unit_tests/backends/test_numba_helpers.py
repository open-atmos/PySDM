from PySDM.backends.numba.numba_helpers import bisec, pair_indices
import os
import numpy as np
import pytest


class TestNumbaHelpers:
    @staticmethod
    def test_bisec():
        # arrange
        sut = bisec if 'NUMBA_DISABLE_JIT' in os.environ else bisec.py_func

        def f(x):
            return x

        # act
        zero = sut(minfun=f, a=-1, interval=2, args=(), rtol=.1)

        # assert
        np.testing.assert_almost_equal(f(zero), 0)

    @staticmethod
    @pytest.mark.parametrize("i, idx, is_first_in_pair, expected", [
        (0, (0, 1), (True, False), (0, 1)),
        (0, (1, 0), (True, False), (1, 0)),
        (0, (0, 1, 2), (False, True), (1, 2)),
    ])
    def test_pair_indices(i, idx, is_first_in_pair, expected):
        # Arrange
        sut = pair_indices if 'NUMBA_DISABLE_JIT' in os.environ else pair_indices.py_func

        # Act
        actual = sut(i, idx, is_first_in_pair)

        # Assert
        assert expected == actual

