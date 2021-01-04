"""
Created at 2020
"""

from PySDM.environments._moist import _Moist
from PySDM.products.product import Product


class DryAirDensity(Product):

    def __init__(self):
        super().__init__(
            description="Dry-air density",
            name="rhod",
            unit="kg/m^3",
            range=(0.95, 1.3),
            scale="linear"
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['rhod'])
        return self.buffer
