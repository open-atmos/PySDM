"""
cloud albedo
"""

from PySDM.products.impl.product import Product
from PySDM.products.size_spectral.cloud_optical_depth import CloudOpticalDepth


class CloudAlbedo(Product):
    def __init__(self, *, unit="None", name=None):
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        pass

    def _impl(self, **kwargs):
        return self.formulae.cloud_albedo.albedo(CloudOpticalDepth())
