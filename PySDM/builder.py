"""
The Builder class handling creation of  `PySDM.particulator.Particulator` instances
"""

import inspect
import warnings

import numpy as np

from PySDM.attributes.impl.attribute_registry import get_attribute_class
from PySDM.impl.particle_attributes_factory import ParticleAttributesFactory
from PySDM.impl.wall_timer import WallTimer
from PySDM.initialisation.discretise_multiplicities import (  # TODO #324
    discretise_multiplicities,
)
from PySDM.particulator import Particulator
from PySDM.physics.particle_shape_and_density import LiquidSpheres, MixedPhaseSpheres


def _warn_env_as_ctor_arg():
    warnings.warn(
        "PySDM > v2.31 Builder expects environment instance as argument",
        DeprecationWarning,
    )


class Builder:
    def __init__(self, n_sd, backend, environment=None):
        assert not inspect.isclass(backend)
        self.formulae = backend.formulae
        self.particulator = Particulator(n_sd, backend)
        self.req_attr_names = ["multiplicity", "water mass", "cell id"]
        self.req_attr = None
        self.aerosol_radius_threshold = 0
        self.condensation_params = None

        if environment is None:
            _warn_env_as_ctor_arg()
        else:
            self._set_environment(environment)

    def _set_condensation_parameters(self, **kwargs):
        self.condensation_params = kwargs

    def set_environment(self, environment):
        _warn_env_as_ctor_arg()
        self._set_environment(environment)

    def _set_environment(self, environment):
        if self.particulator.environment is not None:
            raise AssertionError("environment has already been set")
        self.particulator.environment = environment
        self.particulator.environment.register(self)

    def add_dynamic(self, dynamic):
        assert self.particulator.environment is not None
        key = inspect.getmro(type(dynamic))[-2].__name__
        assert key not in self.particulator.dynamics
        self.particulator.dynamics[key] = dynamic

    def _register_product(self, product, buffer):
        if product.name in self.particulator.products:
            raise ValueError(f'product name "{product.name}" already registered')
        product.set_buffer(buffer)
        product.register(self)
        self.particulator.products[product.name] = product

    def _resolve_attribute(self, attr_name):
        if attr_name not in self.req_attr:
            self.req_attr[attr_name] = get_attribute_class(
                attr_name,
                self.particulator.dynamics.keys(),
                self.formulae,
            )(self)
            assert self.req_attr is not None

    def get_attribute(self, attribute_name):
        """intended for obtaining attribute instances during build() logic,
        from within register() methods"""
        self._resolve_attribute(attribute_name)
        return self.req_attr[attribute_name]

    def request_attribute(self, attribute_name):
        """can be called either before or during build()"""
        if self.req_attr_names is not None:
            self.req_attr_names.append(attribute_name)
        else:
            self._resolve_attribute(attribute_name)

    def build(
        self,
        attributes: dict,
        products: tuple = (),
        int_caster=discretise_multiplicities,
    ):
        assert self.particulator.environment is not None

        if "n" in attributes and "multiplicity" not in attributes:
            attributes["multiplicity"] = attributes["n"]
            del attributes["n"]
            warnings.warn(
                'renaming attributes["n"] to attributes["multiplicity"]',
                DeprecationWarning,
            )

        if "volume" in attributes and "water mass" not in attributes:
            assert self.particulator.formulae.particle_shape_and_density.__name__ in (
                LiquidSpheres.__name__,
                MixedPhaseSpheres.__name__,
            ), "implied volume-to-mass conversion is only supported for spherical particles"
            attributes["water mass"] = (
                self.particulator.formulae.particle_shape_and_density.volume_to_mass(
                    attributes["volume"]
                )
            )
            del attributes["volume"]
            self.request_attribute("volume")

        self.req_attr = {}
        for attr_name in self.req_attr_names:
            self._resolve_attribute(attr_name)
        self.req_attr_names = None

        for dynamic in self.particulator.dynamics.values():
            dynamic.register(self)

        single_buffer_for_all_products = np.empty(self.particulator.mesh.grid)
        for product in products:
            self._register_product(product, single_buffer_for_all_products)

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
        attributes["multiplicity"] = int_caster(attributes["multiplicity"])
        if self.particulator.mesh.dimension == 0:
            attributes["cell id"] = np.zeros_like(
                attributes["multiplicity"], dtype=np.int64
            )
        self.particulator.attributes = ParticleAttributesFactory.attributes(
            self.particulator, self.req_attr, attributes
        )
        self.particulator.recalculate_cell_id()

        for key in self.particulator.dynamics:
            self.particulator.timers[key] = WallTimer()

        return self.particulator
