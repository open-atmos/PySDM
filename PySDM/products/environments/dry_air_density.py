"""
Created at 2020
"""

from PySDM.products.product import MoistEnvironmentProduct


class DryAirDensity(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Dry-air density",
            name="rhod",
            unit="kg/m^3",
            range=(0.95, 1.3),
            scale="linear"
        )

