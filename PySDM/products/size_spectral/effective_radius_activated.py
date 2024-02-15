"""
effective radius of particles within a grid cell, for activated, unactivated or both
"""

import numpy as np

from PySDM.products.impl.activation_filtered_product import _ActivationFilteredProduct
from PySDM.products.impl.moment_product import MomentProduct
from PySDM.products.size_spectral.effective_radius import EffectiveRadius


class ActivatedEffectiveRadius(MomentProduct, _ActivationFilteredProduct):
    def __init__(
        self, *, count_unactivated: bool, count_activated: bool, name=None, unit="m"
    ):
        MomentProduct.__init__(self, name=name, unit=unit)
        _ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        _ActivationFilteredProduct.register(self, builder)
        MomentProduct.register(self, builder)

    def _impl(self, **kwargs):
        _ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=2 / 3,
        )
        tmp = np.empty_like(self.buffer)
        tmp[:] = self.buffer[:]
        _ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=1,
        )
        EffectiveRadius.nan_aware_reff_impl(
            input_volume_output_reff=self.buffer, volume_2_3=tmp
        )
        return self.buffer
