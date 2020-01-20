"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.physics.constants import p1000, Rd, c_pd, rho_w
from PySDM.simulation.physics import formulae as phys
import numpy as np
from PySDM.backends.numba.numba import Numba
import numba

idx_thd = 0
idx_lnv = 1


class _ODESystem:
    def __init__(self, kappa, dry_volume: np.ndarray, n: np.ndarray, dthd_dt, dqv_dt, m_d_mean, rhod_mean, qt):
        self.kappa = kappa
        self.rd = phys.radius(volume=dry_volume)
        self.n = n
        self.dqv_dt = dqv_dt
        self.dthd_dt = dthd_dt
        self.rhod_mean = rhod_mean
        self.m_d_mean = m_d_mean
        self.qt = qt

    def __call__(self, t, y):
        thd = y[idx_thd]
        lnv = y[idx_lnv:]

        qv = self.qt + self.dqv_dt * t - self.ql(self.n, lnv, self.m_d_mean)
        T, p, RH = Numba.temperature_pressure_RH(self.rhod_mean, thd, qv)

        dy_dt = np.empty_like(y)
        self.impl(dy_dt, lnv, T, p, self.n, RH, self.kappa, self.rd, qv, self.dthd_dt, self.dqv_dt, self.m_d_mean)
        return dy_dt

    @staticmethod
    @numba.njit()
    def ql(n, lnv, m_d_mean):
        return np.sum(n * np.exp(lnv)) * rho_w / m_d_mean

    @staticmethod
    @numba.njit()
    def impl(dy_dt, lnv, T, p, n, RH, kappa, rd, qv, dot_thd, dot_qv, m_d_mean):
        for i in range(len(lnv)):
            r = (np.exp(lnv[i]) * 3 / 4 / np.pi) ** (1 / 3)
            dy_dt[idx_lnv + i] = 3/r * phys.dr_dt_MM(r, T, p, RH - 1, kappa, rd[i])

        dqv_dt = dot_qv - np.sum(n * np.exp(lnv) * dy_dt[idx_lnv:]) * rho_w / m_d_mean
        dy_dt[idx_thd] = dot_thd - phys.lv(T) * dqv_dt / phys.c_p(qv) * (p1000 / p) ** (Rd / c_pd)  # TODO: p_d?

    def derr(self, lnv, thd, qv, rd):
        T, p, RH = Numba.temperature_pressure_RH(self.rhod_mean, thd, qv)
        return _ODESystem.derr_impl(lnv, T, p, RH, self.kappa, rd)

    # TODO: move to phys (same above)
    @staticmethod
    @numba.njit()
    def derr_impl(lnv, T, p, RH, kappa, rd):
        r = (np.exp(lnv) * 3 / 4 / np.pi) ** (1 / 3)
        return 3 / r * phys.dr_dt_MM(r, T, p, RH - 1, kappa, rd)
