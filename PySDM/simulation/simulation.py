"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.stats import Stats


# TODO: better name?
class Simulation:

    def __init__(self, n_sd, grid, size, dt, backend):

        self.__n_sd = n_sd
        self.__grid = grid
        self.__size = size
        self.dt = dt
        self.backend = backend()
        self.state: State = None
        self.dynamics: list = []
        self.__dv = None
        self.__dimension = len(grid)

        self.n_steps = 0
        self.stats = Stats()  # TODO: inject?

    @property
    def dimension(self) -> int:
        return self.__dimension

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    @property
    def grid(self) -> tuple:
        return self.__grid

    @property
    def n_cell(self) -> int:
        if self.dimension == 0:
            return 1
        if self.dimension == 2:
            return self.grid[0] * self.grid[1]
        raise NotImplementedError()

    @property
    def size(self) -> tuple:
        return self.__size

    # TODO: hardcoded 2D
    @property
    def dx(self) -> float:
        return self.size[0] / self.grid[0]

    @property
    def dz(self) -> float:
        return self.size[1] / self.grid[1]

    @property
    def dv(self) -> float:
        if self.dimension == 0:
            return self.__dv
        if self.dimension == 2:
            return self.dx * self.dz
        raise NotImplementedError()

    def set_dv(self, value):
        self.__dv = value

    def add_dynamics(self, dynamic_class, params):
        self.dynamics.append(dynamic_class(self, *params))

    def create_state_0d(self, n, intensive, extensive):
        if self.state is None:
            self.state = StateFactory.state_0d(n, intensive, extensive, self)
        else:
            raise AssertionError("State are already initialized.")

    def create_state_2d(self, n, extensive, intensive, positions):
        if self.state is None:
            self.state = StateFactory.state_2d(n, self.grid, intensive, extensive, positions, self)
        else:
            raise AssertionError("State are already initialized.")

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics:
                    dynamic()
        self.n_steps += steps

