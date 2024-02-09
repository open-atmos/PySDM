"""
ambient temperature
"""

from PySDM.products.impl.moist_environment_product import MoistEnvironmentProduct


class AmbientTemperature(MoistEnvironmentProduct):
    def __init__(self, name=None, unit="K", var=None):
        super().__init__(name=name, unit=unit, var=var)
