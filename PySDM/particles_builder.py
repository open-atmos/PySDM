"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.particles import Particles
from PySDM.initialisation.multiplicities import n_init, discretise_n
from PySDM.physics import formulae as phys
from PySDM.state.state_factory import StateFactory
from PySDM.initialisation.r_wet_init import r_wet_init
from PySDM.initialisation.temperature_init import temperature_init

from .state.products.aerosol_concentration import AerosolConcentration
from .state.products.aerosol_specific_concentration import AerosolSpecificConcentration
from .state.products.total_particle_concentration import TotalParticleConcentration
from .state.products.total_particle_specific_concentration import TotalParticleSpecificConcentration
from .state.products.particle_mean_radius import ParticleMeanRadius
from .state.products.super_droplet_count import SuperDropletCount
from .state.products.particle_temperature import ParticleTemperature
from .state.products.particles_size_spectrum import ParticlesSizeSpectrum
from .state.products.particles_volume_spectrum import ParticlesVolumeSpectrum

from PySDM.attributes.mapper import get_class as attr_class
from PySDM.attributes.droplet.multiplicities import Multiplicities
from PySDM.attributes.droplet.volume import Volume
from PySDM.attributes.cell.cell_id import CellID


class ParticlesBuilder:

    def __init__(self, n_sd, backend, stats=None):
        self.particles = Particles(n_sd, backend, stats)
        self.req_attr = {'n': Multiplicities(self), 'volume': Volume(self), 'cell id': CellID(self)}
        self.aerosol_radius_threshold = 0

    def _set_condensation_parameters(self, coord, adaptive=True):
        self.particles.condensation_solver = self.particles.backend.make_condensation_solver(coord, adaptive)

    def set_environment(self, environment_class, params: dict):
        assert_none(self.particles.environment)
        self.particles.environment = environment_class(self, **params)

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

    def get_attribute(self, attribute_name):
        self.request_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute):
        if attribute not in self.req_attr:
            self.req_attr[attribute] = attr_class(attribute)(self)

    def get_particles(self, attributes: dict, products: dict = {}):
        if attributes is None:
            assert_not_none(self.particles.state)
        else:
            for attribute in attributes:
                self.get_attribute(attribute)
            attributes['n'] = discretise_n(attributes['n'])
            if self.particles.mesh.dimension == 0:
                attributes['cell id'] = np.zeros_like(attributes['n'], dtype=np.int64)
            self.particles.state = StateFactory.attributes(self.particles, self.req_attr, attributes)

            for product_class, args in products.items():
                self.register_product(product_class(self, **args))

        return self.particles


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")
