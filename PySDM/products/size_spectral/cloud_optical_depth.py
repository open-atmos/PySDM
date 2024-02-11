"""
cloud optical depth
"""

from PySDM.products.impl.product import Product
from PySDM.products.size_spectral.effective_radius_activated import (
    ActivatedEffectiveRadius,
)
from PySDM.products.displacement.cloud_water_path import CloudWaterPath


class CloudOpticalDepth(Product):
    def __init__(self, *, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        pass

    def _impl(self, **kwargs):
        # todo how to use other products inside this product?
        return self.formulae.optical_depth.tau(
            CloudWaterPath(),
            ActivatedEffectiveRadius(count_unactivated=False, count_activated=True),
        )
