"""
Created at 05.02.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.physics import constants as const
from PySDM.products.product import Product


class WaterVapourMixingRatio(Product):

    def __init__(self):
        super().__init__(
            description="Water vapour mixing ratio",
            name="qv",
            unit="g/kg",
            range=(5, 7.5),
            scale="linear",
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.core.env, _Moist)
        self.environment = builder.core.env

    def get(self):
        self.download_to_buffer(self.environment['qv'])
        const.convert_to(self.buffer, const.si.gram / const.si.kilogram)
        return self.buffer
