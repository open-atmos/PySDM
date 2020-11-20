from PySDM.backends.numba.numba_helpers import bisec
import os
import numpy as np


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