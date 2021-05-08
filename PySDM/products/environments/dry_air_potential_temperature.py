"""
Created at 05.02.2020
"""

from PySDM.environments._moist import _Moist
from PySDM.products.product import MoistEnvironmentProduct


class DryAirPotentialTemperature(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Dry-air potential temperature",
            name="thd",
            unit="K"
        )
