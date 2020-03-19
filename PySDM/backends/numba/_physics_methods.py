"""
Created at 11.2019
"""

import numba
from PySDM.backends.numba import conf
from .numba_helpers import temperature_pressure_RH, radius, dthd_dt, dr_dt_MM, dr_dt_FF


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
    def temperature_pressure_RH(rhod, thd, qv):
        return temperature_pressure_RH(rhod, thd, qv)

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
