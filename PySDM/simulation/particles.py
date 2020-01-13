"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.state.state import State
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.stats import Stats
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
from PySDM.simulation.mesh import Mesh


class Particles:

    def __init__(self, n_sd, dt, backend, stats=None):
        self.__n_sd = n_sd
        self.__dt = dt
        self.backend = backend
        self.mesh = None
        self.environment = None
        self.state: (State, None) = None
        self.dynamics: list = []
        self.__dv = None
        self.n_steps = 0
        self.stats = stats if stats is not None else Stats()

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    @property
    def dt(self) -> float:
        return self.__dt

    def set_mesh(self, grid, size):
        assert_none(self.mesh)
        self.mesh = Mesh(grid, size)

    def set_mesh_0d(self, dv=None):
        assert_none(self.mesh)
        self.mesh = Mesh.mesh_0d(dv)

    def set_environment(self, environment_class, params: dict):
        assert_not_none(self.mesh)
        assert_none(self.environment)
        self.environment = environment_class(self, **params)

    def add_dynamic(self, dynamic_class, params: dict):
        self.dynamics.append(dynamic_class(self, **params))

    def create_state_0d(self, n, extensive, intensive):
        n = discretise_n(n)
        assert_not_none(self.mesh)
        assert_none(self.state)
        cell_id = np.zeros_like(n, dtype=np.int64)
        self.state = StateFactory.state(n=n,
                                        intensive=intensive,
                                        extensive=extensive,
                                        cell_id=cell_id,
                                        cell_origin=None,
                                        position_in_cell=None,
                                        particles=self)

    def create_state_2d(self, extensive, intensive, spatial_discretisation, spectral_discretisation,
                        spectrum_per_mass_of_dry_air, r_range, kappa):
        assert_not_none(self.mesh, self.environment)
        assert_none(self.state)

        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.mesh.grid, self.n_sd)
            cell_id, cell_origin, position_in_cell = self.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation(self.n_sd, spectrum_per_mass_of_dry_air, r_range)
            r_wet = r_wet_init(r_dry, self.environment, cell_id, kappa)
            n_per_m3 = n_init(n_per_kg, self, cell_id)
            n = discretise_n(n_per_m3)

        extensive['volume'] = phys.volume(radius=r_wet)
        extensive['dry volume'] = phys.volume(radius=r_dry)

        self.state = StateFactory.state(n, intensive, extensive, cell_id, cell_origin, position_in_cell, self)

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics:
                    dynamic()
                self.environment.post_step()
        self.n_steps += steps

    ###

    def find_pairs(self, cell_start, is_first_in_pair):
        self.backend.find_pairs(cell_start, is_first_in_pair,
                                self.state.cell_id,
                                self.state.idx,
                                self.state.SD_num)

    def sum_pair(self, output, x, is_first_in_pair):
        self.backend.sum_pair(output, self.state.get_backend_storage(x),
                              is_first_in_pair,
                              self.state.idx,
                              self.state.SD_num)

    def max_pair(self, prob, is_first_in_pair):
        self.backend.max_pair(prob, self.state.n, is_first_in_pair, self.state.idx, self.state.SD_num)

    def normalize(self, prob, cell_start, norm_factor):
        self.backend.normalize(prob, self.state.cell_id, cell_start, norm_factor, self.dt / self.mesh.dv)

    def coalescence(self, gamma):
        self.backend.coalescence(n=self.state.n,
                                 idx=self.state.idx,
                                 length=self.state.SD_num,
                                 intensive=self.state.get_intensive_attrs(),
                                 extensive=self.state.get_extensive_attrs(),
                                 gamma=gamma,
                                 healthy=self.state.healthy)


# TODO: move somewhere
def n_init(n_per_kg, particles, cell_id):
    n_per_m3 = n_per_kg * particles.environment["rhod"][cell_id]
    domain_volume = np.prod(np.array(particles.mesh.size))
    return n_per_m3 * domain_volume


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")


def discretise_n(y_float):
    y_int = y_float.round().astype(np.int64)

    percent_diff = abs(1 - np.sum(y_float) / np.sum(y_int.astype(float)))
    if percent_diff > .01:
        raise Exception(f"{percent_diff}% error in total real-droplet number due to casting multiplicities to ints")

    return y_int
