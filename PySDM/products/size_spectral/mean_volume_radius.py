"""
mean volume radius of particles within a grid cell, for activated, unactivated or both
"""

from PySDM.products.impl import (
    ActivationFilteredProduct,
    MomentProduct,
    register_product,
)


@register_product()
class MeanVolumeRadius(MomentProduct, ActivationFilteredProduct):
    def __init__(
        self, count_unactivated: bool, count_activated: bool, name=None, unit="m"
    ):
        MomentProduct.__init__(self, name=name, unit=unit)
        ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        for base_class in (ActivationFilteredProduct, MomentProduct):
            base_class.register(self, builder)

    def _impl(self, **kwargs):
        ActivationFilteredProduct.impl(self, attr="volume", rank=1)
        return self.formulae.trivia.radius(self.buffer[:])
