"""
Created at 2020
"""

import numba
from PySDM.backends.numba import conf
import numpy as np
from PySDM.physics.formulae import radius


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dx_dt(x, dr_dt):
    r = radius(x)
    return 4 * np.pi * r**2 * dr_dt


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def volume(x):
    return x


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def x(volume):
    return volume

