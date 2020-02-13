"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

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

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics.values():
                    dynamic()
                self.environment.post_step()
        self.n_steps += steps
