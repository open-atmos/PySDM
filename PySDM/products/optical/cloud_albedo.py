"""
cloud albedo
"""

from PySDM.products.impl.product import Product


class CloudAlbedo(Product):
    def __init__(self, *, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        return self.formulae.optical_albedo.albedo(kwargs["optical_depth"])
