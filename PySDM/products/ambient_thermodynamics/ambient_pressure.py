"""
ambient pressure
"""

from PySDM.products.impl.moist_environment_product import MoistEnvironmentProduct


class AmbientPressure(MoistEnvironmentProduct):
    def __init__(self, name=None, unit="Pa", var=None):
        super().__init__(name=name, unit=unit, var=var)
