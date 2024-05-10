"""
cloud water content products, Specific means per mass of dry air

CloudWaterContent is both liquid and ice
LiquidWaterContent is just liquid
IceWaterContent is just ice

"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


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
                attr="water mass", rank=1, filter_range=(-np.inf, 0)
            )
            mass = self.buffer.copy()

            self._download_moment_to_buffer(
                attr="water mass", rank=0, filter_range=(-np.inf, 0)
            )
            conc = self.buffer
            cwc -= mass * conc / self.particulator.mesh.dv

        if self.specific:
            self._download_to_buffer(self.particulator.environment["rhod"])
            cwc /= self.buffer
        return cwc


class SpecificCloudWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=True, ice=True)


class LiquidWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/m^3", name=None):
        super().__init__(unit=unit, name=name, specific=False, liquid=True, ice=False)


class SpecificLiquidWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=True, ice=False)


class IceWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/m^3", name=None):
        super().__init__(unit=unit, name=name, specific=False, liquid=False, ice=True)


class SpecificIceWaterContent(CloudWaterContent):
    def __init__(self, unit="kg/kg", name=None):
        super().__init__(unit=unit, name=name, specific=True, liquid=False, ice=True)
