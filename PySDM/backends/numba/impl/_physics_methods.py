"""
Created at 11.2019
"""

import numba
from numba import prange
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH, radius, dthd_dt, dr_dt_MM, dr_dt_FF


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
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def temperature_pressure_RH_body(rhod, thd, qv, T, p, RH):
        for i in range(T.shape[0]):
            T[i], p[i], RH[i] = temperature_pressure_RH(rhod[i], thd[i], qv[i])

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

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def radius(volume):
        return radius(volume)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dr_dt_MM(r, T, p, RH, kp, rd):
        return dr_dt_MM(r, T, p, RH, kp, rd)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
        return dr_dt_FF(r, T, p, qv, kp, rd, T_i)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dthd_dt(rhod, thd, T, dqv_dt):
        return dthd_dt(rhod, thd, T, dqv_dt)
