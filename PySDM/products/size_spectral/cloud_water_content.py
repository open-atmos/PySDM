"""
cloud water content products
if `specific=True`, reports values per mass of dry air, otherwise per volume

CloudWaterContent is both liquid and ice
LiquidWaterContent is just liquid
IceWaterContent is just ice
"""

import numpy as np

from PySDM.products.impl import MomentProduct, register_product


@register_product()
class CloudWaterContent(MomentProduct):
    def __init__(
        self, unit="kg/m^3", name=None, specific=False, liquid=True, ice=True
    ):  # pylint: disable=too-many-arguments
        super().__init__(unit=unit, name=name)
        self.specific = specific
        self.liquid = liquid
        self.ice = ice

    def _impl(self, **kwargs):
        cwc = 0.0
        if self.liquid:
            self._download_moment_to_buffer(
                attr="water mass", rank=1, filter_range=(0, np.inf)
            )
            mass = self.buffer.copy()

            self._download_moment_to_buffer(
                attr="water mass", rank=0, filter_range=(0, np.inf)
            )
            conc = self.buffer
            cwc += mass * conc / self.particulator.mesh.dv

        if self.ice:
            self._download_moment_to_buffer(
                attr="water mass",
                rank=1,
                filter_range=(-np.inf, 0),
                filter_attr="signed water mass",
            )
            mass = self.buffer.copy()

            self._download_moment_to_buffer(
                attr="water mass",
                rank=0,
                filter_range=(-np.inf, 0),
                filter_attr="signed water mass",
            )
            conc = self.buffer
            cwc += mass * conc / self.particulator.mesh.dv

        if self.specific:
            self._download_to_buffer(self.particulator.environment["rhod"])
            cwc /= self.buffer
        return cwc


@register_product()
class SpecificCloudWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=True, ice=True)


@register_product()
class LiquidWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/m^3", name=None):
        super().__init__(unit=unit, name=name, specific=False, liquid=True, ice=False)


@register_product()
class SpecificLiquidWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=True, ice=False)


@register_product()
class IceWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/m^3", name=None):
        super().__init__(unit=unit, name=name, specific=False, liquid=False, ice=True)


@register_product()
class SpecificIceWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=False, ice=True)
