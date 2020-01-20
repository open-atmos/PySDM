# noinspection PyProtectedMember
from PySDM.backends.numba import _physics_methods as physics
from PySDM.backends.numba.numba import Numba
from PySDM.simulation.physics import constants as const

import numpy as np

backend = Numba()
dr_dt_MM = backend.dr_dt_MM

R = physics.R
r_cr = physics.r_cr
pvs = physics.pvs
lv = physics.lv
c_p = physics.c_p


def th_dry(th_std, qv):
    return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)


def th_std(p, T):
    return T * (const.p1000 / p)**(const.Rd / const.c_pd)


def radius(volume):
    return (volume * 3 / 4 / np.pi) ** (1 / 3)


def volume(radius):
    return 4 / 3 * np.pi * radius ** 3


