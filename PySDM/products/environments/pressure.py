"""
Created at 18.10.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.physics import constants as const
from PySDM.products.product import Product


class Pressure(Product):

    def __init__(self):
        super().__init__(
            description="Pressure",
            name="p_ambient",
            unit="Pa",
            range=(90000, 100000),
            scale="linear",
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['p'])
        return self.buffer
