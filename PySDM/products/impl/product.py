"""
logic around the `PySDM.products.impl.product.Product` - parent class for all products
"""

import inspect
from abc import abstractmethod
from typing import Optional

import pint

from PySDM.physics.constants import PPB, PPM, PPT
from PySDM.impl.camel_case import camel_case_to_words

_UNIT_REGISTRY = pint.UnitRegistry()


class Product:
    def __init__(self, *, unit: str, name: Optional[str] = None):
        self.name = name or camel_case_to_words(self.__class__.__name__)

        self._unit = self._parse_unit(unit)
        self.unit_magnitude_in_base_units = self._unit.to_base_units().magnitude
        self.__check_unit()

        self.shape = None
        self.buffer = None
        self.particulator = None
        self.formulae = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.formulae = self.particulator.formulae
        self.shape = self.particulator.mesh.grid

    def set_buffer(self, buffer):
        self.buffer = buffer

    def _download_to_buffer(self, storage):
        storage.download(self.buffer.ravel())

    @staticmethod
    def _parse_unit(unit: str):
        if unit in ("%", "percent"):
            return 0.01 * _UNIT_REGISTRY.dimensionless
        if unit in ("PPB", "ppb"):
            return PPB * _UNIT_REGISTRY.dimensionless
        if unit in ("PPM", "ppm"):
            return PPM * _UNIT_REGISTRY.dimensionless
        if unit in ("PPT", "ppt"):
            return PPT * _UNIT_REGISTRY.dimensionless
        return _UNIT_REGISTRY.parse_expression(unit)

    def __check_unit(self):
        init = inspect.signature(self.__init__)
        if "unit" not in init.parameters:
            raise AssertionError(
                f"method __init__ of class {type(self).__name__}"
                f" is expected to have a unit parameter"
            )

        default_unit_arg = init.parameters["unit"].default

        if default_unit_arg is None or str(default_unit_arg).strip() == "":
            raise AssertionError(
                f"unit parameter of {type(self).__name__}.__init__"
                f" is expected to have a non-empty default value"
            )

        default_unit = self._parse_unit(default_unit_arg)

        if default_unit.to_base_units().magnitude != 1:
            raise AssertionError(
                f'default value "{default_unit_arg}"'
                f" of {type(self).__name__}.__init__() unit parameter"
                f" is not a base SI unit"
            )

        if self._unit.dimensionality != default_unit.dimensionality:
            raise AssertionError(
                f"provided unit ({self._unit}) has different dimensionality"
                f" ({self._unit.dimensionality}) than the default one"
                f" ({default_unit.dimensionality})"
                f" for product {type(self).__name__}"
            )

    @property
    def unit(self):
        return str(self._unit)

    @abstractmethod
    def _impl(self, **kwargs):
        raise NotImplementedError()

    def get(self, **kwargs):
        result = self._impl(**kwargs)
        result /= self.unit_magnitude_in_base_units
        return result
