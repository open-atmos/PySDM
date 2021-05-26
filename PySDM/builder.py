"""
The Builder class handling creation of  `PySDM.core.Core` instances
"""
import numpy as np

from PySDM.core import Core
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

    def __init__(self, n_sd, backend, formulae=Formulae()):
        assert inspect.isclass(backend)
        self.formulae = formulae
        self.core = Core(n_sd, backend(formulae))
        self.req_attr = {'n': Multiplicities(self), 'volume': Volume(self), 'cell id': CellID(self)}
        self.aerosol_radius_threshold = 0
        self.condensation_params = None

    def _set_condensation_parameters(self, **kwargs):
        self.condensation_params = kwargs

    def set_environment(self, environment):
        assert_none(self.core.environment)
        self.core.environment = environment
        self.core.environment.register(self)

    def add_dynamic(self, dynamic):
        assert_not_none(self.core.environment)
        self.core.dynamics[dynamic.__class__.__name__] = dynamic

    def register_product(self, product):
        if product.name in self.core.products:
            raise Exception(f'product name "{product.name}" already registered')
        product.register(self)
        self.core.products[product.name] = product

    def get_attribute(self, attribute_name):
        self.request_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute):
        if attribute not in self.req_attr:
            self.req_attr[attribute] = attr_class(attribute, self.core.dynamics)(self)

    def build(self, attributes: dict, products: list = (), int_caster=discretise_n):
        self.core.backend.sanity_check()

        for dynamic in self.core.dynamics.values():
            dynamic.register(self)

        for product in products:
            self.register_product(product)

        for attribute in attributes:
            self.request_attribute(attribute)
        if 'Condensation' in self.core.dynamics:
            self.core.condensation_solver = \
                self.core.backend.make_condensation_solver(self.core.dt, self.core.mesh.n_cell, **self.condensation_params)
        attributes['n'] = int_caster(attributes['n'])
        if self.core.mesh.dimension == 0:
            attributes['cell id'] = np.zeros_like(attributes['n'], dtype=np.int64)
        self.core.particles = ParticlesFactory.attributes(self.core, self.req_attr, attributes)

        for key in self.core.dynamics.keys():
            self.core.timers[key] = WallTimer()

        return self.core


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")
