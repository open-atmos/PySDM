"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics.constants import p1000, Rd, c_pd, rho_w

from scipy import integrate as ode

from PySDM.backends.numba.numba import Numba
import numpy as np
import numba

idx_rhod = 0
idx_thd = 1
idx_qv = 2
idx_rw = 3


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
        rw = y[idx_rw:]

        T, p, RH = Numba.temperature_pressure_RH(rhod, thd, qv)

        dy_dt = np.empty_like(y)

        foo(dy_dt, rw, T, p, self.n, RH, self.kappa, self.rd, qv, self.drhod_dt, self.dthd_dt, self.dqv_dt, self.m_d)

        return dy_dt

# TODO
@numba.njit()
def foo(dy_dt, rw, T, p, n, RH, kappa, rd, qv, dot_rhod, dot_thd, dot_qv, m_d):
    dy_dt[idx_qv] = dot_qv
    dy_dt[idx_thd] = dot_thd
    dy_dt[idx_rhod] = dot_rhod

    for i in range(len(rw)):
        dy_dt[idx_rw + i] = phys.dr_dt_MM(rw[i], T, p, RH - 1, kappa, rd[i])
    dy_dt[idx_qv] += -4 * np.pi * np.sum(n * rw ** 2 * dy_dt[idx_rw:]) * rho_w / m_d
    dy_dt[idx_thd] += - phys.lv(T) * dy_dt[idx_qv] / phys.c_p(qv) * (p1000 / p) ** (Rd / c_pd)


def compute_cell_start(cell_start, cell_id, idx, sd_num):
    cell_start[:] = -1

    for i in range(sd_num - 1, -1, -1):  # reversed
        cell_start[cell_id[idx[i]]] = i
    cell_start[-1] = sd_num

    for i in range(len(cell_start) - 1, -1, -1):  # reversed
        if cell_start[i] == -1:
            cell_start[i] = cell_start[i + 1]


class Condensation:
    def __init__(self, particles, kappa):

        self.particles = particles
        self.environment = particles.environment

        self.dt = self.particles.dt
        self.kappa = kappa

        self.rd = None

        self.cell_start = particles.backend.array(self.particles.mesh.n_cell + 1, dtype=int)

        self.scheme = 'scipy.odeint'  # TODO

    # TODO: assumes sorted by cell_id (i.e., executed after coalescence)
    def __call__(self):
        self.environment.sync()

        state = self.particles.state
        state.sort_by_cell_id()  # TODO +what about droplets that precipitated out of the domain
        compute_cell_start(self.cell_start, state.cell_id, state.idx, state.SD_num)

        v = state.get_backend_storage("volume")
        n = state.n
        vdry = state.get_backend_storage("dry volume")

        if self.scheme == 'scipy.odeint':
            for cell_id in range(self.particles.mesh.n_cell):
                cell_start = self.cell_start[cell_id]
                cell_end = self.cell_start[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                y0 = np.empty(n_sd_in_cell + idx_rw)
                y0[idx_rhod] = self.environment['rhod'][cell_id]
                y0[idx_thd] = self.environment['thd'][cell_id]
                y0[idx_qv] = self.environment['qv'][cell_id]
                y0[idx_rw:] = phys.radius(volume=v[state.idx[cell_start:cell_end]])
                integ = ode.solve_ivp(
                    _ODESystem(
                        self.kappa,
                        vdry[state.idx[cell_start:cell_end]],
                        n[state.idx[cell_start:cell_end]],
                        (self.environment.get_predicted("rhod")[cell_id] - self.environment['rhod'][cell_id]) / self.dt,
                        (self.environment.get_predicted('thd')[cell_id] - self.environment['thd'][cell_id]) / self.dt,
                        (self.environment.get_predicted('qv')[cell_id] - self.environment['qv'][cell_id]) / self.dt,
                        self.environment.get_predicted("rhod")[cell_id] * self.environment.dv
                    ),
                    (0., self.dt),
                    y0,
                    method='BDF',
                    # rtol=1e-6,
                    atol=1e-9,
                    # first_step=self.dt,
                    t_eval=[self.dt]
                )
                assert integ.success, integ.message

                dm = 0
                for i in range(cell_end - cell_start):
                    x_new = phys.volume(radius=integ.y[idx_rw + i])
                    x_old = v[state.idx[cell_start + i]]
                    nd = n[state.idx[cell_start + i]]
                    dm += nd * (x_new - x_old) * rho_w
                    v[state.idx[cell_start + i]] = x_new

                m_d = self.environment.get_predicted('rhod')[cell_id] * self.environment.dv
                dq_sum = - dm / m_d
                dq_ode = integ.y[idx_qv] - self.environment.get_predicted('qv')[cell_id]

                #dth_sum =
                dth_ode = integ.y[idx_thd] - self.environment.get_predicted('thd')[cell_id]

                # TODO: move to a separate test
                #np.testing.assert_approx_equal(dq_ode, dq_sum, 4)
                #np.testing.assert_approx_equal(dth_ode, dth_sum)

                self.environment.get_predicted('qv')[cell_id] += dq_sum
                self.environment.get_predicted('thd')[cell_id] += dth_ode
        else:
            raise NotImplementedError()



