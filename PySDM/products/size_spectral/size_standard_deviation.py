"""
standard deviation of radius/area/volume of particles within a grid cell,
for activated, unactivated or both
"""

import numpy as np

from PySDM.products.impl import (
    ActivationFilteredProduct,
    MomentProduct,
    register_product,
)


class _SizeStandardDeviation(MomentProduct, ActivationFilteredProduct):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        count_unactivated: bool,
        count_activated: bool,
        name=None,
        unit="m",
        attr="radius",
    ):
        self.attr = attr
        MomentProduct.__init__(self, name=name, unit=unit)
        ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        builder.request_attribute(self.attr)
        for base_class in (ActivationFilteredProduct, MomentProduct):
            base_class.register(self, builder)

    def _impl(self, **kwargs):
        ActivationFilteredProduct.impl(self, attr=self.attr, rank=1)
        tmp = np.empty_like(self.buffer)
        tmp[:] = -self.buffer**2
        ActivationFilteredProduct.impl(self, attr=self.attr, rank=2)
        tmp[:] += self.buffer
        tmp[:] = np.sqrt(tmp)
        return tmp


RadiusStandardDeviation = register_product()(_SizeStandardDeviation)


@register_product()
class AreaStandardDeviation(_SizeStandardDeviation):
    def __init__(
        self, *, name=None, unit="m^2", count_activated: bool, count_unactivated: bool
    ):
        super().__init__(
            name=name,
            unit=unit,
            count_activated=count_activated,
            count_unactivated=count_unactivated,
            attr="area",
        )


@register_product()
class VolumeStandardDeviation(_SizeStandardDeviation):
    def __init__(
        self, *, name=None, unit="m^3", count_activated: bool, count_unactivated: bool
    ):
        super().__init__(
            name=name,
            unit=unit,
            count_activated=count_activated,
            count_unactivated=count_unactivated,
            attr="volume",
        )
