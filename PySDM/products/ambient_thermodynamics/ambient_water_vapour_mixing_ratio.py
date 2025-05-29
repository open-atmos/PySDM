"""
ambient water vapour mixing ratio (mass of water vapour per mass of dry air)
"""

from PySDM.products.impl import MoistEnvironmentProduct, register_product


@register_product()
class AmbientWaterVapourMixingRatio(MoistEnvironmentProduct):
    def __init__(self, name=None, unit="dimensionless", var=None):
        super().__init__(unit=unit, name=name, var=var)
