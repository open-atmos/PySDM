"""
The Builder class handling creation of  `PySDM.particulator.Particulator` instances
"""
import inspect

import numpy as np

from PySDM.attributes.impl.mapper import get_class as attr_class
from PySDM.attributes.numerics.cell_id import CellID
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.physics.volume import Volume
from PySDM.impl.particle_attributes_factory import ParticleAttributesFactory
from PySDM.impl.wall_timer import WallTimer
from PySDM.initialisation.discretise_multiplicities import (  # TODO #324
    discretise_multiplicities,
)
from PySDM.particulator import Particulator


class Builder:
    def __init__(self, n_sd, backend):
        assert not inspect.isclass(backend)
        self.formulae = backend.formulae
        self.particulator = Particulator(n_sd, backend)
        self.req_attr = {
            "n": Multiplicities(self),
            "volume": Volume(self),
            "cell id": CellID(self),
        }
        self.aerosol_radius_threshold = 0
        self.condensation_params = None

    def _set_condensation_parameters(self, **kwargs):
        self.condensation_params = kwargs

    def set_environment(self, environment):
        if self.particulator.environment is not None:
            raise AssertionError("environment has already been set")
        self.particulator.environment = environment
        self.particulator.environment.register(self)

    def add_dynamic(self, dynamic):
        assert self.particulator.environment is not None
        key = inspect.getmro(type(dynamic))[-2].__name__
        assert key not in self.particulator.dynamics
        self.particulator.dynamics[key] = dynamic

    def register_product(self, product, buffer):
        if product.name in self.particulator.products:
            raise Exception(f'product name "{product.name}" already registered')
        product.set_buffer(buffer)
        product.register(self)
        self.particulator.products[product.name] = product

    def get_attribute(self, attribute_name):
        self.request_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute, variant=None):
        if attribute not in self.req_attr:
            self.req_attr[attribute] = attr_class(
                attribute, self.particulator.dynamics
            )(self)
        if variant is not None:
            assert variant == self.req_attr[attribute]

    def build(
        self,
        attributes: dict,
        products: tuple = (),
        int_caster=discretise_multiplicities,
    ):
        assert self.particulator.environment is not None

        for dynamic in self.particulator.dynamics.values():
            dynamic.register(self)

        single_buffer_for_all_products = np.empty(self.particulator.mesh.grid)
        for product in products:
            self.register_product(product, single_buffer_for_all_products)

        for attribute in attributes:
            self.request_attribute(attribute)
        if "Condensation" in self.particulator.dynamics:
            self.particulator.condensation_solver = (
                self.particulator.backend.make_condensation_solver(
                    self.particulator.dt,
                    self.particulator.mesh.n_cell,
                    **self.condensation_params,
                )
            )
        attributes["n"] = int_caster(attributes["n"])
        if self.particulator.mesh.dimension == 0:
            attributes["cell id"] = np.zeros_like(attributes["n"], dtype=np.int64)
        self.particulator.attributes = ParticleAttributesFactory.attributes(
            self.particulator, self.req_attr, attributes
        )
        self.particulator.recalculate_cell_id()

        for key in self.particulator.dynamics:
            self.particulator.timers[key] = WallTimer()

        return self.particulator
