"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.utils import Physics
from PySDM.simulation.physics.formulae import dr_dt_MM, lv, c_p
from PySDM.simulation.physics.constants import p1000, Rd, c_pd, rho_w

from scipy import integrate as ode

from PySDM.backends.numba.numba import Numba
import numpy as np
import numba

idx_qv = 0  # redundant
idx_thd = 1
idx_rw = 2


class _ODESystem:
    def __init__(self, rhod, kappa, xd: np.ndarray, n: np.ndarray, dqv_dt, dthd_dt, m_d):
        self.rhod = rhod
        self.kappa = kappa
        self.rd = Physics.x2r(xd)
        self.n = n  # TODO: per mass of dry air !
        self.dqv_dt = dqv_dt
        self.dthd_dt = dthd_dt
        self.m_d = m_d

    def __call__(self, t, y):
        thd = y[idx_thd]
        qv = y[idx_qv]
        rw = y[idx_rw:]

        T, p, RH = Numba.temperature_pressure_RH(self.rhod, thd, qv)

        dy_dt = np.empty_like(y)

        foo(dy_dt, rw, T, p, self.n, RH, self.kappa, self.rd, qv, self.dqv_dt, self.dthd_dt, self.m_d)

        return dy_dt

# TODO !!! (incl. np.minimum())
@numba.njit()
def foo(dy_dt, rw, T, p, n, RH, kappa, rd, qv, dqv_dt, dthd_dt, m_d):
    for i in range(len(rw)):
        dy_dt[idx_rw + i] = dr_dt_MM(rw[i], T, p, np.minimum(RH - 1, .01), kappa, rd[i])
    dy_dt[idx_qv] = -4 * np.pi * np.sum(n * rw ** 2 * dy_dt[idx_rw:]) * rho_w / m_d
    dy_dt[idx_thd] = - lv(T) * dy_dt[idx_qv] / c_p(qv) * (p1000 / p) ** (Rd / c_pd)

    dy_dt[idx_qv] += dqv_dt
    dy_dt[idx_thd] += dthd_dt


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
        state.sort_by_cell_id() #TODO
        compute_cell_start(self.cell_start, state.cell_id, state.idx, state.SD_num)

        x = state.get_backend_storage("x")
        n = state.n
        xdry = state.get_backend_storage("dry volume")

        if self.scheme == 'scipy.odeint':
            for cell_id in range(self.particles.mesh.n_cell):
                cell_start = self.cell_start[cell_id]
                cell_end = self.cell_start[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                y0 = np.empty(n_sd_in_cell + 2)
                y0[idx_thd] = self.environment['thd'][cell_id]
                y0[idx_qv] = self.environment['qv'][cell_id]
                y0[idx_rw:] = Physics.x2r(x[state.idx[cell_start:cell_end]])
                integ = ode.solve_ivp(
                    _ODESystem(
                        self.environment.get_predicted("rhod")[cell_id],
                        self.kappa,
                        xdry[state.idx[cell_start:cell_end]],
                        n[state.idx[cell_start:cell_end]],
                        (self.environment.get_predicted('qv')[cell_id] - self.environment['qv'][cell_id]) / self.dt,
                        (self.environment.get_predicted('thd')[cell_id] - self.environment['thd'][cell_id]) / self.dt,
                        self.environment.get_predicted("rhod")[cell_id] * self.environment.dv
                    ),
                    (0., self.dt),
                    y0,
                    method='BDF',
                    # rtol=1e-6,
                    atol=1e-8,
                    # first_step=self.dt,
                    t_eval=[self.dt]
                )
                assert integ.success, integ.message

                dm = 0
                for i in range(cell_end - cell_start):
                    x_new = Physics.r2x(integ.y[idx_rw + i])
                    x_old = x[state.idx[cell_start + i]]
                    nd = n[state.idx[cell_start + i]]
                    dm += nd * (x_new - x_old) * rho_w
                    x[state.idx[cell_start + i]] = x_new

                m_d = self.environment.get_predicted('rhod')[cell_id] * self.environment.dv
                dq_sum = - dm / m_d
                dq_ode = integ.y[idx_qv] - self.environment.get_predicted('qv')[cell_id]

                #dth_sum =
                dth_ode = integ.y[idx_thd] - self.environment.get_predicted('thd')[cell_id]

                #np.testing.assert_approx_equal(dq_ode, dq_sum, 4)
                #np.testing.assert_approx_equal(dth_ode, dth_sum)

                self.environment.get_predicted('qv')[cell_id] += dq_sum
                self.environment.get_predicted('thd')[cell_id] += dth_ode
        else:
            raise NotImplementedError()

        # TODO: what about droplets that precipitated out of the domain


