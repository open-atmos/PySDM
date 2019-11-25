"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.environment.kinematic_2d import Kinematic2D
from PySDM.utils import Physics
from PySDM.simulation.physics.formulae import dr_dt_MM, lv, c_p
from PySDM.simulation.physics.constants import p1000, Rd, c_pd

from scipy import integrate as ode

from PySDM.backends.numba.numba import Numba
import numpy as np
import numba

idx_qv = 0  # redundant
idx_thd = 1
idx_rw = 2


class _ODESystem:
    def __init__(self, rhod, kappa, xd: np.ndarray, n: np.ndarray, dqv_dt, dthd_dt):
        self.rhod = rhod
        self.kappa = kappa
        self.rd = Physics.x2r(xd)
        self.n = n  # TODO: per mass of dry air !
        self.rho_w = 1  # TODO
        self.dqv_dt = dqv_dt
        self.dthd_dt = dthd_dt
        # self.dy_dt = np.empty(len(n) + 2)

    def __call__(self, t, y):
        thd = y[idx_thd]
        qv = y[idx_qv]
        rw = y[idx_rw:]

        T, p, RH = Numba.temperature_pressure_RH(self.rhod, thd, qv)

        dy_dt = np.empty_like(y)

        foo(dy_dt, rw, T, p, self.n, RH, self.kappa, self.rd, self.rho_w, qv, self.dqv_dt, self.dthd_dt)

        return dy_dt

# TODO
@numba.njit()
def foo(dy_dt, rw, T, p, n, RH, kappa, rd, rho_w, qv, dqv_dt, dthd_dt):
    for i in range(len(rw)):
        dy_dt[idx_rw + i] = dr_dt_MM(rw[i], T, p, np.minimum(RH - 1, .01), kappa, rd[i])
    dy_dt[idx_qv] = -4 * np.pi * np.sum(n * rw ** 2 * dy_dt[idx_rw:]) * rho_w
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
    def __init__(self, particles, environment, kappa):

        self.particles = particles
        self.environment = environment

        self.dt = environment.dt
        self.kappa = kappa

        self.rd = None

        self.cell_start = particles.backend.array(environment.n_cell + 1, dtype=int)

        self.scheme = 'scipy.odeint'  # TODO

    # TODO: assumes sorted by cell_id (i.e., executed after coalescence)
    def __call__(self):
        state = self.particles.state
        state.sort_by_cell_id() #TODO

        self.environment.sync()
        new = self.environment['new']
        old = self.environment['old']

        compute_cell_start(self.cell_start, state.cell_id, state.idx, state.SD_num)
        # print(self.cell_start)

        x = state.get_backend_storage("x")
        n = state.n
        xdry = state.get_backend_storage("dry volume")

        if self.scheme == 'scipy.odeint':
            for cell_id in range(self.environment.n_cell):
                cell_start = self.cell_start[cell_id]
                cell_end = self.cell_start[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                # print(old['RH'][cell_id], " -> ", new['RH'][cell_id]) TODO

                y0 = np.empty(n_sd_in_cell + 2)
                y0[idx_thd] = old['thd'][cell_id]
                y0[idx_qv] = old['qv'][cell_id]
                y0[idx_rw:] = Physics.x2r(x[state.idx[cell_start:cell_end]])
                integ = ode.solve_ivp(
                    _ODESystem(
                        self.environment.rhod[cell_id],
                        self.kappa,
                        xdry[state.idx[cell_start:cell_end]],
                        n[state.idx[cell_start:cell_end]],
                        (new['qv'][cell_id] - old['qv'][cell_id]) / self.dt,
                        (new['thd'][cell_id] - old['thd'][cell_id]) / self.dt
                    ),
                    (0., self.dt),
                    y0,
                    method='BDF',
                    rtol=1e-6,
                    atol=1e-22,
#                    first_step=self.dt,
                    t_eval=[self.dt]
                )
                assert integ.success, integ.message

                for i in range(cell_end - cell_start):
                    # print(x[state.idx[cell_start + i]], Physics.r2x(integ.y[idx_rw + i]))
                    x[state.idx[cell_start + i]] = Physics.r2x(integ.y[idx_rw + i])
                new['qv'][cell_id] = integ.y[idx_qv]
                new['thd'][cell_id] = integ.y[idx_thd]
                # print()
                # TODO: RH_new, T_new, p_new
        else:
            raise NotImplementedError()

        # TODO: what about droplets that precipitated out of the domain


