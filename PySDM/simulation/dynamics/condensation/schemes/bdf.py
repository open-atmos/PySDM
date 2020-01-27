"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.physics.constants import rho_w
from PySDM.simulation.physics import formulae as phys
import numpy as np
import numba
import scipy.integrate

idx_thd = 0
idx_lnv = 1


def impl(y,
         v, n, vdry,
         cell_idx,
         dt, kappa,
         thd, qv,
         dthd_dt, dqv_dt,
         m_d_mean, rhod_mean,
         rtol, atol, dt_max
         ):
    n_sd_in_cell = len(cell_idx)
    y0 = y[0:n_sd_in_cell + idx_lnv]
    y0[idx_thd] = thd
    y0[idx_lnv:] = np.log(v[cell_idx])  # TODO: abstract out ln()
    qt = qv + _ODESystem.ql(n[cell_idx], y0[idx_lnv:], m_d_mean)

    integ = scipy.integrate.solve_ivp(
        _ODESystem(
            kappa,
            vdry[cell_idx],
            n[cell_idx],
            dthd_dt,
            dqv_dt,
            m_d_mean,
            rhod_mean,
            qt
        ),
        t_span=[0, dt],
        t_eval=[dt],
        y0=y0,
        rtol=rtol,
        atol=atol,
        method="BDF"
    )
    assert integ.success, integ.message
    y1 = integ.y[:, 0]

    m_new = 0
    for i in range(n_sd_in_cell):
        x_new = np.exp(y1[idx_lnv + i])
        m_new += n[cell_idx[i]] * x_new * rho_w
        v[cell_idx[i]] = x_new

    return qt - m_new/m_d_mean, y1[idx_thd]


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
        T, p, RH = phys.temperature_pressure_RH(self.rhod_mean, thd, qv)

        dy_dt = np.empty_like(y)
        self.impl(dy_dt, lnv, T, p, self.n, RH, self.kappa, self.rd, thd, self.dthd_dt, self.dqv_dt, self.m_d_mean, self.rhod_mean)
        return dy_dt

    @staticmethod
    @numba.njit()
    def ql(n, lnv, m_d_mean):
        return np.sum(n * np.exp(lnv)) * rho_w / m_d_mean

    @staticmethod
    @numba.njit()
    def impl(dy_dt, lnv, T, p, n, RH, kappa, rd, thd, dot_thd, dot_qv, m_d_mean, rhod_mean):
        for i in range(len(lnv)):
            dy_dt[idx_lnv + i] = phys.dlnv_dt(lnv[i], T, p, RH, kappa, rd[i])
        dqv_dt = dot_qv - np.sum(n * np.exp(lnv) * dy_dt[idx_lnv:]) * rho_w / m_d_mean
        dy_dt[idx_thd] = dot_thd + phys.dthd_dt(rhod_mean, thd, T, dqv_dt)
