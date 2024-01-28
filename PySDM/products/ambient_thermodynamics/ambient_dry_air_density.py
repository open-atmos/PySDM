"""
ambient dry-air density
"""

from PySDM.products.impl.moist_environment_product import MoistEnvironmentProduct


class AmbientDryAirDensity(MoistEnvironmentProduct):
    def __init__(self, name="rhod", unit="kg/m^3", var=None):
        super().__init__(name=name, unit=unit, var=var)
