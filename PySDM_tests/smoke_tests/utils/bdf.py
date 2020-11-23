"""
Created at 09.01.2020
"""

from PySDM.physics.constants import rho_w
from PySDM.physics import formulae as phys
from PySDM.backends.numba.numba import Numba
from PySDM.backends.numba.coordinates import volume as coord_volume
from PySDM.backends.numba.coordinates import volume_logarithm as coord_volume_logarithm
import numpy as np
import numba
import scipy.integrate
import types

idx_thd = 0
idx_x = 1


def patch_core(core, coord='volume logarithm', rtol=1e-3):
    core.condensation_solver = make_solve(coord, rtol)
    core.condensation = types.MethodType(bdf_condensation, core)


def bdf_condensation(core,
                     kappa,
                     rtol_x, rtol_thd, substeps, ripening_flags
                     ):
    n_threads = 1
    if core.particles.has_attribute("temperature"):
        raise NotImplementedError()

    Numba._condensation.py_func(
        solver=core.condensation_solver,
        n_threads=n_threads,
        n_cell=core.mesh.n_cell,
        cell_start_arg=core.particles.cell_start.data,
        v=core.particles["volume"].data,
        particle_temperatures=np.empty(0),
        r_cr=None,
        n=core.particles['n'].data,
        vdry=core.particles["dry volume"].data,
        idx=core.particles._Particles__idx.data,
        rhod=core.env["rhod"].data,
        thd=core.env["thd"].data,
        qv=core.env["qv"].data,
        dv_mean=core.env.dv,
        prhod=core.env.get_predicted("rhod").data,
        pthd=core.env.get_predicted("thd").data,
        pqv=core.env.get_predicted("qv").data,
        kappa=kappa,
        rtol_x=rtol_x,
        rtol_thd=rtol_thd,
        dt=core.dt,
        substeps=substeps.data,
        cell_order=np.argsort(substeps),
        ripening_flags=ripening_flags.data
    )


def make_solve(coord, rtol):
    if coord == 'volume':
        coord = coord_volume
    elif coord == 'volume logarithm':
        coord = coord_volume_logarithm
    else:
        raise ValueError()

    x = coord.x
    volume = coord.volume
    dx_dt = coord.dx_dt

    def solve(
            v, particle_temperatures, r_cr, n, vdry,
            cell_idx, kappa, thd, qv,
            dthd_dt, dqv_dt, m_d_mean, rhod_mean,
            rtol_x, rtol_thd, dt, substeps
    ):
        n_sd_in_cell = len(cell_idx)
        y0 = np.empty(n_sd_in_cell + idx_x)
        y0[idx_thd] = thd
        y0[idx_x:] = x(v[cell_idx])
        qt = qv + _ODESystem.ql(n[cell_idx], y0[idx_x:], m_d_mean)

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
            atol=0,
            method="BDF"
        )
        assert integ.success, integ.message
        y1 = integ.y[:, 0]

        m_new = 0
        for i in range(n_sd_in_cell):
            v_new = volume(y1[idx_x + i])
            m_new += n[cell_idx[i]] * v_new * rho_w
            v[cell_idx[i]] = v_new

        return qt - m_new / m_d_mean, y1[idx_thd], 1, 1  # TODO: how to get the number of timesteps?

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
            x = y[idx_x:]

            qv = self.qt + self.dqv_dt * t - self.ql(self.n, x, self.m_d_mean)
            T, p, RH = phys.temperature_pressure_RH(self.rhod_mean, thd, qv)

            dy_dt = np.empty_like(y)
            self.impl(dy_dt, x, T, p, self.n, RH, self.kappa, self.rd, thd, self.dthd_dt, self.dqv_dt, self.m_d_mean,
                      self.rhod_mean)
            return dy_dt

        @staticmethod
        @numba.njit()
        def ql(n, x, m_d_mean):
            return np.sum(n * volume(x)) * rho_w / m_d_mean

        @staticmethod
        @numba.njit()
        def impl(dy_dt, x, T, p, n, RH, kappa, rd, thd, dot_thd, dot_qv, m_d_mean, rhod_mean):
            for i in range(len(x)):
                dy_dt[idx_x + i] = dx_dt(x[i], phys.dr_dt_MM(phys.radius(volume(x[i])), T, p, RH, kappa, rd[i]))
            dqv_dt = dot_qv - np.sum(n * volume(x) * dy_dt[idx_x:]) * rho_w / m_d_mean
            dy_dt[idx_thd] = dot_thd + phys.dthd_dt(rhod_mean, thd, T, dqv_dt)

    return solve
