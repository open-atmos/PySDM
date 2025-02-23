"""
ambient pressure
"""

from PySDM.products.impl import MoistEnvironmentProduct, register_product


@register_product()
class AmbientPressure(MoistEnvironmentProduct):
    def __init__(self, name=None, unit="Pa", var=None):
        super().__init__(name=name, unit=unit, var=var)
