# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import numba
import numpy as np
import pytest
from scipy.optimize import toms748

from PySDM.backends.impl_numba.toms748 import toms748_solve
from PySDM.formulae import Formulae

# relevant
# https://github.com/scipy/scipy/blob/master/scipy/optimize/tests/test_zeros.py
# https://github.com/boostorg/math/blob/develop/test/test_toms748_solve.cpp


@numba.njit()
def f1(x):
    return x**2 - 2 * x - 1


@numba.njit()
def f2(x):
    return np.exp(x) - np.cos(x)


@pytest.mark.parametrize("fun", (f1, f2))
def test_toms748(fun):
    sut = toms748_solve if "NUMBA_DISABLE_JIT" in os.environ else toms748_solve.py_func

    a = -0.5
    b = 0.5
    rtol = 1e-6
    wt = Formulae().trivia.within_tolerance
    actual, _ = sut(
        fun,
        (),
        ax=a,
        bx=b,
        fax=fun(a),
        fbx=fun(b),
        max_iter=10,
        rtol=rtol,
        within_tolerance=wt,
    )
    expected = toms748(fun, a, b)

    np.testing.assert_almost_equal(actual, expected)
