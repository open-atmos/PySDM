"""
Created at 11.2019
"""

import numba
from numba import prange
from PySDM.backends.numba import conf
from PySDM.physics.formulae import temperature_pressure_pv, pvs


class PhysicsMethods:
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def explicit_in_space(omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def implicit_in_space(omega, c_l, c_r):
        """
        see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        """
        dC = c_r - c_l
        return (omega * dC + c_l) / (1 - dC)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def temperature_pressure_RH_body(rhod, thd, qv, T, p, RH):
        for i in prange(T.shape[0]):
            T[i], p[i], pv = temperature_pressure_pv(rhod[i], thd[i], qv[i])
            RH[i] = pv / pvs(T[i])

    @staticmethod
    def temperature_pressure_RH(rhod, thd, qv, T, p, RH):
        return PhysicsMethods.temperature_pressure_RH_body(rhod.data, thd.data, qv.data, T.data, p.data, RH.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def terminal_velocity(values, radius, k1, k2, k3, r1, r2):
        for i in prange(len(values)):
            if radius[i] < r1:
                values[i] = k1 * radius[i] ** 2
            elif radius[i] < r2:
                values[i] = k2 * radius[i]
            else:
                values[i] = k3 * radius[i] ** (1 / 2)


