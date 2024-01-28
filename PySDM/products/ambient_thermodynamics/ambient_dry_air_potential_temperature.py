"""
ambient dry-air potential temperature (computed using dry air partial pressure)
"""

from PySDM.products.impl.moist_environment_product import MoistEnvironmentProduct


class AmbientDryAirPotentialTemperature(MoistEnvironmentProduct):
    def __init__(self, unit="K", name=None, var=None):
        super().__init__(unit=unit, name=name, var=var)
