"""
Created at 18.10.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.physics import constants as const
from PySDM.products.product import Product


class Temperature(Product):

    def __init__(self):
        super().__init__(
            description="Temperature",
            name="T_ambient",
            unit="K",
            range=(275,305),
            scale="linear",
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['T'])
        return self.buffer
