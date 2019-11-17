"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation.environment.moist_air import MoistAir
from PySDM.utils import Physics
from PySDM.simulation.physics.formulae import dr_dt_MM, lv, c_p
from PySDM.simulation.physics.constants import p1000, Rd, c_pd

from scipy import integrate as ode

from PySDM.backends.numba.numba import Numba
import numpy as np

idx_qv = 0  # redundant
idx_thd = 1
idx_rw = 2


class _ODESystem:
    def __init__(self, rhod, kappa, rd: np.ndarray, n: np.ndarray):
        self.rhod = rhod
        self.kappa = kappa
        self.rd = rd
        self.n = n # TODO: per mass of dry air !
        self.rho_w = 1 # TODO

    def __call__(self, t, y):
        thd = y[idx_thd]
        qv = y[idx_qv]
        rw = y[idx_rw:]

        T, p, RH = Numba.temperature_pressure_RH(self.rhod, thd, qv)

        dy_dt = np.empty_like(y)
        for i in range(len(rw)):
            dy_dt[idx_rw + i] = dr_dt_MM(rw[i], T, p, RH-1, self.kappa, self.rd[i])
        dy_dt[idx_qv] = -4 * np.pi * np.sum(self.n * rw**2 * dy_dt[idx_rw:]) * self.rho_w
        dy_dt[idx_thd] = - lv(T) * dy_dt[idx_qv] / c_p(qv) * (p1000/p) ** (Rd/c_pd)

        return dy_dt


def compute_cell_start(cell_start, cell_id, idx, sd_num):
    for i in range(sd_num - 1, -1, -1):  # reversed
        cell_start[cell_id[idx[i]]] = i
    cell_start[-1] = sd_num


class Condensation:
    def __init__(self, ambient_air: MoistAir, dt, kappa, backend, n_cell):
        self.ambient_air = ambient_air

        self.dt = dt
        self.kappa = kappa

        self.rd = None

        self.cell_start = backend.array(n_cell + 1, dtype=int)
        self.n_cell = n_cell

        self.scheme = 'scipy.odeint'

    # TODO: assumes sorted by cell_id (i.e., executed after coalescence)
    def __call__(self, state: State):
        state.sort_by_cell_id()
        self.ambient_air.sync()

        compute_cell_start(self.cell_start, state.cell_id, state.idx, state.SD_num)

        x = state.get_backend_storage("x")
        n = state.n
        rdry = state.get_backend_storage("dry radius")

        if self.scheme == 'explicitEuler':
            for i in state.idx[:state.SD_num]:
                cid = state.cell_id[i]
                r = Physics.x2r(x[i])
                T = self.ambient_air.T[cid]
                p = self.ambient_air.p[cid]
                S = (self.ambient_air.RH[cid] - 1)
                kp = self.kappa
                rd = rdry[i]

                # explicit Euler
                r_new = r + self.dt * state.backend.dr_dt_MM(r, T, p, S, kp, rd)

                x[i] = Physics.r2x(r_new)
        elif self.scheme == 'scipy.odeint':
            for cell_id in reversed(range(self.n_cell)):
                cell_start = self.cell_start[cell_id]
                cell_end = self.cell_start[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue
                y0 = np.empty(n_sd_in_cell + 2)
                y0[idx_thd] = self.ambient_air.thd[cell_id]
                y0[idx_qv] = self.ambient_air.qv[cell_id]
                y0[idx_rw:] = Physics.x2r(x[state.idx[cell_start:cell_end]])
                integ = ode.solve_ivp(
                    _ODESystem(
                        self.ambient_air.rhod[cell_id],
                        self.kappa,
                        rdry[state.idx[cell_start:cell_end]],
                        n[state.idx[cell_start:cell_end]]
                    ),
                    (0., self.dt),
                    y0,
                    method='BDF',
                    rtol=1e-6,
                    atol=1e-6,
                    first_step=self.dt/10,
                    t_eval=[self.dt]
                )
                assert integ.success, integ.message
                # TODO: qv, thd
                for i in range(cell_end - cell_start):
                    x[state.idx[cell_start + i]] = Physics.r2x(integ.y[idx_rw + i])
        else:
            raise NotImplementedError()

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain


