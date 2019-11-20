"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM import utils
from PySDM.simulation.state.state import State
from PySDM.simulation.environment.moist_air import MoistAir
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.stats import Stats
from PySDM.simulation.initialisation.r_wet_init import r_wet_init


class Particles:

    def __init__(self, n_sd, grid, size, dt, backend):

        self.__n_sd = n_sd
        self.__grid = grid
        self.__size = size
        self.dt = dt
        self.backend = backend()
        self.state: (State, None) = None
        self.environment: (MoistAir, None) = None
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

    def set_environment(self, environment_class, params):
        self.environment = environment_class(self, *params)

    def add_dynamics(self, dynamic_class, params):
        self.dynamics.append(dynamic_class(self, *params))

    # TODO: extensive, intensive
    def create_state_0d(self, n, intensive, extensive):
        if self.state is None:
            self.state = StateFactory.state_0d(n, intensive, extensive, self)
        else:
            raise AssertionError("State is already initialized.")

    def create_state_2d(self, n, extensive, intensive, positions):
        if self.state is None:
            self.state = StateFactory.state_2d(n, self.grid, intensive, extensive, positions, self)
        else:
            raise AssertionError("State is already initialized.")

    def create_state_2d2(self, extensive, intensive, spatial_discretisation, spectral_discretisation,
                         spectrum_per_mass_of_dry_air, r_range, kappa):
        if self.environment is None:
            raise AssertionError("Environment is not initialized.")

        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.grid, self.n_sd)

            r_dry, n_per_kg = spectral_discretisation(
                self.n_sd, spectrum_per_mass_of_dry_air, r_range
            )
            # TODO: cell_id, _, _ = StateFactory.positions(n_per_kg, positions)  # TODO

            cell_origin = positions.astype(dtype=int)
            strides = utils.strides(self.grid)
            cell_id = np.dot(strides, cell_origin.T).ravel()
            # </TEMP>

            # TODO: not here
            n_per_m3 = n_per_kg * self.environment.rhod[cell_id]
            domain_volume = np.prod(np.array(self.size))
            n = (n_per_m3 * domain_volume).astype(np.int64)
            r_wet = r_wet_init(r_dry, self.environment, cell_id, kappa)

        extensive['x'] = utils.Physics.r2x(r_wet)  # TODO: rename x -> ...
        extensive['dry volume'] = utils.Physics.r2x(r_dry)

        self.create_state_2d(n, extensive, intensive, positions)

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics:
                    dynamic()
        self.n_steps += steps

