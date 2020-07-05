"""
Created at 09.11.2019
"""

import numpy as np

from PySDM.core import Core
from PySDM.initialisation.multiplicities import discretise_n  # TODO
from PySDM.state.state_factory import StateFactory

from PySDM.attributes.mapper import get_class as attr_class
from PySDM.attributes.droplet.multiplicities import Multiplicities
from PySDM.attributes.droplet.volume import Volume
from PySDM.attributes.cell.cell_id import CellID


class Builder:

    def __init__(self, n_sd, backend, stats=None):
        self.core = Core(n_sd, backend, stats)
        self.req_attr = {'n': Multiplicities(self), 'volume': Volume(self), 'cell id': CellID(self)}
        self.aerosol_radius_threshold = 0
        self.condensation_params = None

    def _set_condensation_parameters(self, coord, adaptive=True):
        self.condensation_params = {'coord': coord, 'adaptive': adaptive}

    def set_environment(self, environment_class, params: dict):
        assert_none(self.core.environment)
        self.core.environment = environment_class(self, **params)

    def register_dynamic(self, dynamic_class, params: dict):
        instance = (dynamic_class(self, **params))
        self.core.dynamics[str(dynamic_class)] = instance

    def register_product(self, product):
        if product.name in self.core.products:
            raise Exception(f'product name "{product.name}" already registered')
        self.core.products[product.name] = product

    def get_attribute(self, attribute_name):
        self.request_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute):
        if attribute not in self.req_attr:
            self.req_attr[attribute] = attr_class(attribute)(self)

    def get_particles(self, attributes: dict, products: dict = {}):
        for attribute in attributes:
            self.request_attribute(attribute)
        if "<class 'PySDM.dynamics.condensation.condensation.Condensation'>" in self.core.dynamics:  # TODO: mapper?
            self.core.condensation_solver = \
                self.core.backend.make_condensation_solver(**self.condensation_params,
                                                           enable_drop_temperatures='temperatures' in self.req_attr)
        attributes['n'] = discretise_n(attributes['n'])
        if self.core.mesh.dimension == 0:
            attributes['cell id'] = np.zeros_like(attributes['n'], dtype=np.int64)  # TODO
        self.core.state = StateFactory.attributes(self.core, self.req_attr, attributes)

        for product_class, args in products.items():
            self.register_product(product_class(self, **args))

        return self.core


def assert_none(*params):
    for param in params:
        if param is not None:
            raise AssertionError(str(param.__class__.__name__) + " is already initialized.")


def assert_not_none(*params):
    for param in params:
        if param is None:
            raise AssertionError(str(param.__class__.__name__) + " is not initialized.")
