"""
cloud albedo
"""

from PySDM.products.impl import Product, register_product


@register_product()
class CloudAlbedo(Product):
    def __init__(self, *, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        return self.formulae.optical_albedo.albedo(kwargs["optical_depth"])
