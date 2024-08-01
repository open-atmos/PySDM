"""
cloud optical depth
"""

from PySDM.products.impl import Product, register_product


@register_product()
class CloudOpticalDepth(Product):
    def __init__(self, *, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        return self.formulae.optical_depth.tau(
            kwargs["liquid_water_path"],
            kwargs["effective_radius"],
        )
