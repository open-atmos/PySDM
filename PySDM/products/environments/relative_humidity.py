"""
Created at 05.02.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.products.product import Product


class RelativeHumidity(Product):

    def __init__(self):
        super().__init__(
            description="Relative humidity",
            name="RH",
            unit="%",
            range=(75, 105),
            scale="linear",
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['RH'])
        self.buffer *= 100
        return self.buffer
