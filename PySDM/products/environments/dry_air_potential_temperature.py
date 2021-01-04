"""
Created at 05.02.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.products.product import Product


class DryAirPotentialTemperature(Product):

    def __init__(self):
        super().__init__(
            description="Dry-air potential temperature",
            name="thd",
            unit="K",
            range=(275, 300),
            scale="linear",
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['thd'])
        return self.buffer
