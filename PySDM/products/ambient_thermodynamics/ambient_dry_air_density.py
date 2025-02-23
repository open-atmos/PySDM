"""
ambient dry-air density
"""

from PySDM.products.impl import MoistEnvironmentProduct, register_product


@register_product()
class AmbientDryAirDensity(MoistEnvironmentProduct):
    def __init__(self, name="rhod", unit="kg/m^3", var=None):
        super().__init__(name=name, unit=unit, var=var)
