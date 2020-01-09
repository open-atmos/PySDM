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

idx_rhod = 0
idx_thd = 1
idx_qv = 2
idx_lnv = 3


class _ODESystem:
    def __init__(self, kappa, xd: np.ndarray, n: np.ndarray, drhod_dt, dthd_dt, dqv_dt, m_d):
        self.kappa = kappa
        self.rd = phys.radius(volume=xd)
        self.n = n
        self.dqv_dt = dqv_dt
        self.dthd_dt = dthd_dt
        self.drhod_dt = drhod_dt
        self.m_d = m_d

    def __call__(self, t, y):
        rhod = y[idx_rhod]
        thd = y[idx_thd]
        qv = y[idx_qv]
        lnv = y[idx_lnv:]

        T, p, RH = Numba.temperature_pressure_RH(rhod, thd, qv)

        dy_dt = np.empty_like(y)

        self.impl(dy_dt, lnv, T, p, self.n, RH, self.kappa, self.rd, qv, self.drhod_dt, self.dthd_dt, self.dqv_dt, self.m_d)
        return dy_dt

    @staticmethod
    @numba.njit()
    def impl(dy_dt, lnv, T, p, n, RH, kappa, rd, qv, dot_rhod, dot_thd, dot_qv, m_d):
        dy_dt[idx_qv] = dot_qv
        dy_dt[idx_thd] = dot_thd
        dy_dt[idx_rhod] = dot_rhod

        for i in range(len(lnv)):
            r = (np.exp(lnv[i]) * 3 / 4 / np.pi) ** (1 / 3)
            dy_dt[idx_lnv + i] = 3/r * phys.dr_dt_MM(r, T, p, RH - 1, kappa, rd[i])
        dy_dt[idx_qv] -= np.sum(n * np.exp(lnv) * dy_dt[idx_lnv:]) * rho_w / m_d
        dy_dt[idx_thd] -= phys.lv(T) * dy_dt[idx_qv] / phys.c_p(qv) * (p1000 / p) ** (Rd / c_pd)
