"""
Numba njit-ted basic arithmetics routines for CPU backend
"""

import numba
from PySDM.backends.impl_numba import conf


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def multiply(output, multiplier):
    output *= multiplier
