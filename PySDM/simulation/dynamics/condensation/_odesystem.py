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
idx_qv = 1
idx_lnv = 2


class _ODESystem:
    def __init__(self, kappa, dry_volume: np.ndarray, n: np.ndarray, dthd_dt, dqv_dt, m_d_mean, rhod_mean):
        self.kappa = kappa
        self.rd = phys.radius(volume=dry_volume)
        self.n = n
        self.dqv_dt = dqv_dt
        self.dthd_dt = dthd_dt
        self.rhod_mean = rhod_mean
        self.m_d_mean = m_d_mean

    def __call__(self, _, y):
        thd = y[idx_thd]
        qv = y[idx_qv]
        lnv = y[idx_lnv:]

        T, p, RH = Numba.temperature_pressure_RH(self.rhod_mean, thd, qv)

        dy_dt = np.empty_like(y)

        self.impl(dy_dt, lnv, T, p, self.n, RH, self.kappa, self.rd, qv, self.dthd_dt, self.dqv_dt, self.m_d_mean)
        return dy_dt

    @staticmethod
    @numba.njit()
    def impl(dy_dt, lnv, T, p, n, RH, kappa, rd, qv, dot_thd, dot_qv, m_d_mean):
        dy_dt[idx_qv] = dot_qv
        dy_dt[idx_thd] = dot_thd

        for i in range(len(lnv)):
            r = (np.exp(lnv[i]) * 3 / 4 / np.pi) ** (1 / 3)
            dy_dt[idx_lnv + i] = 3/r * phys.dr_dt_MM(r, T, p, RH - 1, kappa, rd[i])

        dy_dt[idx_qv] -= np.sum(n * np.exp(lnv) * dy_dt[idx_lnv:]) * rho_w / m_d_mean
        dy_dt[idx_thd] -= phys.lv(T) * dy_dt[idx_qv] / phys.c_p(qv) * (p1000 / p) ** (Rd / c_pd)

    def derr(self, lnv, thd, qv, rd):
        T, p, RH = Numba.temperature_pressure_RH(self.rhod_mean, thd, qv)
        return _ODESystem.derr_impl(lnv, T, p, RH, self.kappa, rd)


    @staticmethod
    @numba.njit()
    def derr_impl(lnv, T, p, RH, kappa, rd):
        r = (np.exp(lnv) * 3 / 4 / np.pi) ** (1 / 3)
        return 3 / r * phys.dr_dt_MM(r, T, p, RH - 1, kappa, rd)
