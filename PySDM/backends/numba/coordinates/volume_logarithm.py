"""
Created at 2020
"""

import numba
import numpy as np
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import radius

volume0 = 1.


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dx_dt(x, dr_dt):
    r = radius(volume(x))
    return 3 / r * dr_dt


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def volume(x):
    return volume0 * np.exp(x)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def x(volume):
    return np.log(volume / volume0)
