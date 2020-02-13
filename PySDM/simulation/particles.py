"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.initialisation.multiplicities import n_init, discretise_n
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.state.state import State
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.stats import Stats
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
from PySDM.simulation.mesh import Mesh
from PySDM.simulation.terminal_velocity import TerminalVelocity

from .state.products.aerosol_concentration import AerosolConcentration
from .state.products.aerosol_specific_concentration import AerosolSpecificConcentration
from .state.products.total_particle_concentration import TotalParticleConcentration
from .state.products.particle_mean_radius import ParticleMeanRadius
from .state.products.super_droplet_count import SuperDropletCount
from .state.products.total_particle_specific_concentration import TotalParticleSpecificConcentration

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
        self.register_products(self.environment)

    def register_dynamic(self, dynamic_class, params: dict):
        instance = (dynamic_class(self, **params))
        self.dynamics[str(dynamic_class)] = instance
        self.register_products(instance)

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
                        spectrum_per_mass_of_dry_air, r_range, kappa, radius_threshold):
        assert_not_none(self.mesh, self.environment)
        assert_none(self.state)

        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.mesh.grid, self.n_sd)
            cell_id, cell_origin, position_in_cell = self.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation(self.n_sd, spectrum_per_mass_of_dry_air, r_range)
            r_wet = r_wet_init(r_dry, self.environment, cell_id, kappa)
            n_per_m3 = n_init(n_per_kg, self.environment, self.mesh, cell_id)
            n = discretise_n(n_per_m3)

        extensive['volume'] = phys.volume(radius=r_wet)
        extensive['dry volume'] = phys.volume(radius=r_dry)

        self.state = StateFactory.state(n, intensive, extensive, cell_id, cell_origin, position_in_cell, self)
        for product in [
            TotalParticleConcentration(self),
            TotalParticleSpecificConcentration(self),
            AerosolConcentration(self, radius_threshold),
            AerosolSpecificConcentration(self, radius_threshold),
            ParticleMeanRadius(self),
            SuperDropletCount(self)
        ]:
            self.register_product(product)

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics.values():
                    dynamic()
                self.environment.post_step()
        self.n_steps += steps

    ###

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

    def register_products(self, instance):
        if hasattr(instance, 'products'):
            for product in instance.products:
                self.register_product(product)

    def register_product(self, product):
        name = product.name.lower()
        if name in self.products:
            raise Exception(f"product name >>{product.name}<< already registered (case insensitive)")
        self.products[name] = product


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")
