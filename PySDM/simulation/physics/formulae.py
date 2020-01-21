# noinspection PyProtectedMember
from PySDM.backends.numba import _physics_methods as physics
from PySDM.simulation.physics import constants as const

import numpy as np

dr_dt_MM = physics.dr_dt_MM
R = physics.R
r_cr = physics.r_cr
pvs = physics.pvs
lv = physics.lv
c_p = physics.c_p
dlnv_dt = physics.dlnv_dt
dthd_dt = physics.dthd_dt
temperature_pressure_RH = physics.temperature_pressure_RH
radius = physics.radius


def th_dry(th_std, qv):
    return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)


def th_std(p, T):
    return T * (const.p1000 / p)**(const.Rd / const.c_pd)

def volume(radius):
    return 4 / 3 * np.pi * radius ** 3


