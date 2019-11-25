"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM import utils
from PySDM.simulation.state.state import State
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.stats import Stats
from PySDM.simulation.initialisation.r_wet_init import r_wet_init


class Particles:

    def __init__(self, n_sd, backend):

        self.__n_sd = n_sd
        self.backend = backend()
        self.state: (State, None) = None
        self.environment = None
        self.dynamics: list = []
        self.__dv = None

        self.n_steps = 0
        self.stats = Stats()  # TODO: inject?

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    # TODO use params: dict
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
            self.state = StateFactory.state_2d(n, self.environment.grid, intensive, extensive, positions, self)
        else:
            raise AssertionError("State is already initialized.")

    def create_state_2d2(self, extensive, intensive, spatial_discretisation, spectral_discretisation,
                         spectrum_per_mass_of_dry_air, r_range, kappa):
        if self.environment is None:
            raise AssertionError("Environment is not initialized.")

        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.environment.grid, self.n_sd)

            r_dry, n_per_kg = spectral_discretisation(
                self.n_sd, spectrum_per_mass_of_dry_air, r_range
            )
            # TODO: cell_id, _, _ = StateFactory.positions(n_per_kg, positions)  # TODO

            cell_origin = positions.astype(dtype=int)
            strides = utils.strides(self.environment.grid)
            cell_id = np.dot(strides, cell_origin.T).ravel()
            # </TEMP>

            # TODO: not here
            n_per_m3 = n_per_kg * self.environment.rhod[cell_id]
            domain_volume = np.prod(np.array(self.environment.size))
            n = (n_per_m3 * domain_volume).astype(np.int64)
            r_wet = r_wet_init(r_dry, self.environment['old'], cell_id, kappa)

        extensive['x'] = utils.Physics.r2x(r_wet)  # TODO: rename x -> ...
        extensive['dry volume'] = utils.Physics.r2x(r_dry)

        self.create_state_2d(n, extensive, intensive, positions)

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                self.environment.ante_step()
                for dynamic in self.dynamics:
                    dynamic()
                self.environment.post_step()
        self.n_steps += steps

