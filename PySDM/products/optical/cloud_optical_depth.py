"""
cloud optical depth
"""

from PySDM.products.impl.product import Product


class CloudOpticalDepth(Product):
    def __init__(self, *, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        return self.formulae.optical_depth.tau(
            kwargs["liquid_water_path"],
            kwargs["effective_radius"],
        )
