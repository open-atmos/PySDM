"""
Crated at 2019
"""

from PySDM.backends.numba import numba_helpers as physics
from PySDM.physics import constants as const

import numpy as np

dr_dt_MM = physics.dr_dt_MM
R = physics.R
r_cr = physics.r_cr
pvs = physics.pvs
lv = physics.lv
c_p = physics.c_p
dthd_dt = physics.dthd_dt
temperature_pressure_RH = physics.temperature_pressure_RH
radius = physics.radius
volume = physics.volume
RH_eq = physics.RH_eq
pH2H = physics.pH2H


def th_dry(th_std, qv):
    return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)


def th_std(p, T):
    return T * (const.p1000 / p)**(const.Rd / const.c_pd)


class MoistAir:
    @staticmethod
    def rhod_of_rho_qv(rho, qv):
        return rho / (1 + qv)

    @staticmethod
    def rho_of_rhod_qv(rhod, qv):
        return rhod * (1 + qv)

    @staticmethod
    def p_d(p, qv):
        return p * (1 - 1 / (1 + const.eps / qv))

    @staticmethod
    def rhod_of_pd_T(pd, T):
        return pd / const.Rd / T

    @staticmethod
    def rho_of_p_qv_T(p, qv, T):
        return p / R(qv) / T


class Trivia:
    @staticmethod
    def volume_of_density_mass(rho, m):
        return m / rho


class ThStd:
    @staticmethod
    def rho_d(p, qv, theta_std):
        kappa = const.Rd / const.c_pd
        pd = MoistAir.p_d(p, qv)
        rho_d = pd / (np.power(p / const.p1000, kappa) * const.Rd * theta_std)
        return rho_d


class Hydrostatic:
    @staticmethod
    def drho_dz(g, p, T, qv, dql_dz=0):
        rho = MoistAir.rho_of_p_qv_T(p, qv, T)
        Rq = R(qv)
        cp = c_p(qv)
        return (g / T * rho * (Rq / cp - 1) - p * lv(T) / cp / T**2 * dql_dz) / Rq

    @staticmethod
    def p_of_z_assuming_const_th_and_qv(g, p0, thstd, qv, z):
        kappa = const.Rd / const.c_pd
        z0 = 0
        arg = np.power(p0/const.p1000, kappa) - (z-z0) * kappa * g / thstd / R(qv)
        return const.p1000 * np.power(arg, 1/kappa)


def explicit_euler(y, dt, dy_dt):
    y += dt * dy_dt


def mole_fraction_2_mixing_ratio(mole_fraction, specific_gravity):
    return specific_gravity * mole_fraction / (1 - mole_fraction)


def mixing_ratio_2_mole_fraction(mixing_ratio, specific_gravity):
    return mixing_ratio / (specific_gravity + mixing_ratio)


def mixing_ratio_2_partial_pressure(mixing_ratio, specific_gravity, pressure):
    return pressure * mixing_ratio / (mixing_ratio + specific_gravity)
