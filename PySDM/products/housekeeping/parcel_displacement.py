"""
parcel displacement, for use with `PySDM.environments.parcel.Parcel` environment only
"""

from PySDM.environments import Parcel
from PySDM.products.impl.product import Product


class ParcelDisplacement(Product):
    def __init__(self, unit="m", name=None):
        super().__init__(unit=unit, name=name)
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.particulator.environment, Parcel)
        self.environment = builder.particulator.environment

    def _impl(self, **kwargs):
        self._download_to_buffer(self.environment["z"])
        return self.buffer
