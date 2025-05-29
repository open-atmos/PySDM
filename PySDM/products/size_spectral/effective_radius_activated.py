"""
effective radius of particles within a grid cell, for activated, unactivated or both
"""

import numpy as np

from PySDM.products.impl import (
    ActivationFilteredProduct,
    MomentProduct,
    register_product,
)
from PySDM.products.size_spectral.effective_radius import EffectiveRadius


@register_product()
class ActivatedEffectiveRadius(MomentProduct, ActivationFilteredProduct):
    def __init__(
        self, *, count_unactivated: bool, count_activated: bool, name=None, unit="m"
    ):
        MomentProduct.__init__(self, name=name, unit=unit)
        ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        ActivationFilteredProduct.register(self, builder)
        MomentProduct.register(self, builder)

    def _impl(self, **kwargs):
        ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=2 / 3,
        )
        tmp = np.empty_like(self.buffer)
        tmp[:] = self.buffer[:]
        ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=1,
        )
        EffectiveRadius.nan_aware_reff_impl(
            input_volume_output_reff=self.buffer, volume_2_3=tmp
        )
        return self.buffer
