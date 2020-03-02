"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.state.state import State
from PySDM.simulation.stats import Stats
from PySDM.simulation.terminal_velocity import TerminalVelocity


class Particles:

    def __init__(self, n_sd, dt, backend, stats=None):
        self.__n_sd = n_sd
        self.__dt = dt

        self.backend = backend
        self.mesh = None
        self.environment = None
        self.state: (State, None) = None
        self.dynamics = {}
        self.products = {}

        self.__dv = None
        self.n_steps = 0
        self.stats = stats or Stats()

        self.croupier = 'local'
        self.terminal_velocity = TerminalVelocity(self)
        self.condensation_solver = self.backend.make_condensation_solver()

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    @property
    def dt(self) -> float:
        return self.__dt

    def permute(self, u01):
        if self.croupier == 'global':
            self.state.permutation_global(u01)
        elif self.croupier == 'local':
            self.state.permutation_local(u01)
        else:
            raise NotImplementedError()

    def normalize(self, prob, norm_factor):
        self.backend.normalize(prob, self.state.cell_id, self.state.cell_start, norm_factor, self.dt / self.mesh.dv)

    def find_pairs(self, cell_start, is_first_in_pair):
        self.state.find_pairs(cell_start, is_first_in_pair)

    def sum_pair(self, output, x, is_first_in_pair):
        self.state.sum_pair(output, x, is_first_in_pair)

    def max_pair(self, prob, is_first_in_pair):
        self.state.max_pair(prob, is_first_in_pair)

    def coalescence(self, gamma):
        self.state.coalescence(gamma)

    def remove_precipitated(self):
        self.state.remove_precipitated()

    def condensation(self, kappa, rtol_x, rtol_thd, substeps):
        particle_temperatures = \
            self.state.get_backend_storage("temperature") if self.state.has_attribute("temperature") else \
            self.backend.array(0, dtype=float)

        self.backend.condensation(
                solver=self.condensation_solver,
                n_cell=self.mesh.n_cell,
                cell_start_arg=self.state.cell_start,
                v=self.state.get_backend_storage("volume"),
                particle_temperatures=particle_temperatures,
                n=self.state.n,
                vdry=self.state.get_backend_storage("dry volume"),
                idx=self.state._State__idx,
                rhod=self.environment["rhod"],
                thd=self.environment["thd"],
                qv=self.environment["qv"],
                dv=self.environment.dv,
                prhod=self.environment.get_predicted("rhod"),
                pthd=self.environment.get_predicted("thd"),
                pqv=self.environment.get_predicted("qv"),
                kappa=kappa,
                rtol_x=rtol_x,
                rtol_thd=rtol_thd,
                dt=self.dt,
                substeps=substeps,
                cell_order=np.argsort(substeps)
            )

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics.values():
                    dynamic()
                self.environment.post_step()
        self.n_steps += steps
