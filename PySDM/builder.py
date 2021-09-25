"""
The Builder class handling creation of  `PySDM.Particulator` instances
"""
import numpy as np

from PySDM.particulator import Particulator
from PySDM.initialisation.multiplicities import discretise_n  # TODO #324
from PySDM.state.particles_factory import ParticlesFactory
from PySDM.state.wall_timer import WallTimer
from PySDM.attributes.impl.mapper import get_class as attr_class
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.physics.volume import Volume
from PySDM.attributes.numerics.cell_id import CellID
from PySDM.physics.formulae import Formulae
import inspect


class Builder:

    def __init__(self, n_sd, backend):
        assert not inspect.isclass(backend)
        self.formulae = backend.formulae
        self.particulator = Particulator(n_sd, backend)
        self.req_attr = {'n': Multiplicities(self), 'volume': Volume(self), 'cell id': CellID(self)}
        self.aerosol_radius_threshold = 0
        self.condensation_params = None

    def _set_condensation_parameters(self, **kwargs):
        self.condensation_params = kwargs

    def set_environment(self, environment):
        assert_none(self.particulator.environment)
        self.particulator.environment = environment
        self.particulator.environment.register(self)

    def add_dynamic(self, dynamic):
        assert self.particulator.environment is not None
        self.particulator.dynamics[dynamic.__class__.__name__] = dynamic

    def register_product(self, product):
        if product.name in self.particulator.products:
            raise Exception(f'product name "{product.name}" already registered')
        product.register(self)
        self.particulator.products[product.name] = product

    def get_attribute(self, attribute_name):
        self.request_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute, variant=None):
        if attribute not in self.req_attr:
            self.req_attr[attribute] = attr_class(attribute, self.particulator.dynamics)(self)
        if variant is not None:
            assert variant == self.req_attr[attribute]

    def build(self, attributes: dict, products: list = (), int_caster=discretise_n):
        assert self.particulator.environment is not None

        for dynamic in self.particulator.dynamics.values():
            dynamic.register(self)

        for product in products:
            self.register_product(product)

        for attribute in attributes:
            self.request_attribute(attribute)
        if 'Condensation' in self.particulator.dynamics:
            self.particulator.condensation_solver = \
                self.particulator.backend.make_condensation_solver(self.particulator.dt, self.particulator.mesh.n_cell, **self.condensation_params)
        attributes['n'] = int_caster(attributes['n'])
        if self.particulator.mesh.dimension == 0:
            attributes['cell id'] = np.zeros_like(attributes['n'], dtype=np.int64)
        self.particulator.attributes = ParticlesFactory.attributes(self.particulator, self.req_attr, attributes)

        for key in self.particulator.dynamics:
            self.particulator.timers[key] = WallTimer()

        return self.particulator


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")
