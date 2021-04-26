from PySDM.backends.numba.toms748 import toms748_solve
from scipy.optimize.zeros import toms748
from PySDM.physics.formulae import Formulae
import numba
import numpy as np
import os
import pytest

# relevant
# https://github.com/scipy/scipy/blob/master/scipy/optimize/tests/test_zeros.py
# https://github.com/boostorg/math/blob/develop/test/test_toms748_solve.cpp


@numba.njit()
def f1(x):
    return x ** 2 - 2 * x - 1


@numba.njit()
def f2(x):
    return np.exp(x) - np.cos(x)


@pytest.mark.parametrize("fun", (f1, f2))
def test_toms748(fun):
    sut = toms748_solve if 'NUMBA_DISABLE_JIT' in os.environ else toms748_solve.py_func

    a = -.5
    b = .5
    rtol = 1e-6
    wt = Formulae().trivia.within_tolerance
    actual, iters = sut(f2, (), ax=a, bx=b, fax=f2(a), fbx=f2(b), max_iter=10, rtol=rtol, within_tolerance=wt)
    expected = toms748(f2, a, b)

    np.testing.assert_almost_equal(actual, expected)
