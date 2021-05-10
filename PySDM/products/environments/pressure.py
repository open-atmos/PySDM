"""
Created at 18.10.2020
"""

from PySDM.products.product import MoistEnvironmentProduct


class Pressure(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Pressure",
            name="p",
            unit="Pa"
        )
