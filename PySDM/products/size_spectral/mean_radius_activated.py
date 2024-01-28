"""
mean radius of particles within a grid cell, for activated, unactivated or both
"""

from PySDM.products.impl.activation_filtered_product import _ActivationFilteredProduct
from PySDM.products.impl.moment_product import MomentProduct


class ActivatedMeanRadius(MomentProduct, _ActivationFilteredProduct):
    def __init__(
        self, count_unactivated: bool, count_activated: bool, name=None, unit="m"
    ):
        MomentProduct.__init__(self, name=name, unit=unit)
        _ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        for base_class in (_ActivationFilteredProduct, MomentProduct):
            base_class.register(self, builder)

    def _impl(self, **kwargs):
        _ActivationFilteredProduct.impl(self, attr="volume", rank=1 / 3)
        self.buffer[:] /= self.formulae.constants.PI_4_3 ** (1 / 3)
        return self.buffer
