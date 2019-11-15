# noinspection PyProtectedMember
from PySDM.backends.numba import _physics_methods as physics
from PySDM.backends.numba.numba import Numba


backend = Numba()


class Formulae:
    R = physics.R
    r_cr = physics.r_cr
    dr_dt_MM = backend.dr_dt_MM
    th_dry = physics.th_dry
    pvs = physics.pvs
    lv = physics.lv
    c_p = physics.c_p
