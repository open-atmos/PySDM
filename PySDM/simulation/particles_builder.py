"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles import Particles
from PySDM.simulation.initialisation.multiplicities import n_init, discretise_n
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
from PySDM.simulation.initialisation.temperature_init import temperature_init

from .state.products.aerosol_concentration import AerosolConcentration
from .state.products.aerosol_specific_concentration import AerosolSpecificConcentration
from .state.products.total_particle_concentration import TotalParticleConcentration
from .state.products.total_particle_specific_concentration import TotalParticleSpecificConcentration
from .state.products.particle_mean_radius import ParticleMeanRadius
from .state.products.super_droplet_count import SuperDropletCount
from .state.products.particle_temperature import ParticleTemperature
from .state.products.particles_size_spectrum import ParticlesSizeSpectrum
from .state.products.particles_volume_spectrum import ParticlesVolumeSpectrum


class ParticlesBuilder:

    def __init__(self, n_sd, dt, backend, stats=None):
        self.particles = Particles(n_sd, dt, backend, stats)

    def set_terminal_velocity(self, terminal_velocity_class):
        assert_none(self.particles.terminal_velocity)
        self.particles.terminal_velocity = terminal_velocity_class(self.particles)

    def _set_condensation_parameters(self, coord, adaptive=True):
        self.particles.condensation_solver = self.particles.backend.make_condensation_solver(coord, adaptive)

    def set_environment(self, environment_class, params: dict):
        assert_none(self.particles.environment)
        self.particles.environment = environment_class(self.particles, **params)
        self.register_products(self.particles.environment)

    def register_dynamic(self, dynamic_class, params: dict):
        instance = (dynamic_class(self, **params))
        self.particles.dynamics[str(dynamic_class)] = instance
        self.register_products(instance)

    def register_products(self, instance):
        if hasattr(instance, 'products'):
            for product in instance.products:
                self.register_product(product)

    def register_product(self, product):
        if product.name in self.particles.products:
            raise Exception(f'product name "{product.name}" already registered')
        self.particles.products[product.name] = product

    def set_attributes(self, params):
        assert_not_none(self.particles.environment)
        assert_none(self.particles.state)
        if self.particles.mesh.dimension == 0:
            self.create_state_0d(**params)
        elif self.particles.mesh.dimension == 2:
            self.create_state_2d(**params)
        else:
            NotImplementedError("Implemented dimensions: [0D, 2D]")

    def create_state_0d(self, n, extensive, intensive):
        n = discretise_n(n)
        cell_id = np.zeros_like(n, dtype=np.int64)
        self.particles.state = StateFactory.state(n=n,
                                                  intensive=intensive,
                                                  extensive=extensive,
                                                  cell_id=cell_id,
                                                  cell_origin=None,
                                                  position_in_cell=None,
                                                  particles=self.particles)

        products = [
            TotalParticleConcentration(self.particles),
            TotalParticleSpecificConcentration(self.particles),
            ParticleMeanRadius(self.particles),
            SuperDropletCount(self.particles),
            ParticlesSizeSpectrum(self.particles),
            ParticlesVolumeSpectrum(self.particles)
        ]

        for product in products:
            self.register_product(product)

    def create_state_2d(self, extensive, intensive, spatial_discretisation, spectral_discretisation,
                        spectrum_per_mass_of_dry_air, r_range, kappa, radius_threshold,
                        enable_temperatures: bool = False):

        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.particles.mesh.grid, self.particles.n_sd)
            cell_id, cell_origin, position_in_cell = self.particles.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation(self.particles.n_sd, spectrum_per_mass_of_dry_air, r_range)
            r_wet = r_wet_init(r_dry, self.particles.environment, cell_id, kappa)
            n_per_m3 = n_init(n_per_kg, self.particles.environment, self.particles.mesh, cell_id)
            n = discretise_n(n_per_m3)

        if enable_temperatures:
            T_i = temperature_init(self.particles.environment, cell_id)
            intensive['temperature'] = T_i
            self.register_product(ParticleTemperature(self.particles))

        extensive['volume'] = phys.volume(radius=r_wet)
        extensive['dry volume'] = phys.volume(radius=r_dry)

        self.particles.state = StateFactory.state(n, intensive, extensive,
                                                  cell_id, cell_origin, position_in_cell,
                                                  self.particles)
        products = [
            TotalParticleConcentration(self.particles),
            TotalParticleSpecificConcentration(self.particles),
            AerosolConcentration(self.particles, radius_threshold),
            AerosolSpecificConcentration(self.particles, radius_threshold),
            ParticleMeanRadius(self.particles),
            SuperDropletCount(self.particles)
        ]

        for product in products:
            self.register_product(product)

    def get_particles(self):
        assert_not_none(self.particles.state)
        return self.particles


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")
