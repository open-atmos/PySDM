"""
cloud water path in parcel environment
calculated by summing up cloud water content at each timestep
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class ParcelCloudWaterPath(MomentProduct):
    def __init__(self, name=None, unit="kg/m^2", liquid=True, ice=True):
        super().__init__(unit=unit, name=name)
        self.liquid = liquid
        self.ice = ice
        self.environment = None
        self.previous_z = None
        self._reset_counters()

    def _reset_counters(self):
        self.previous_z = 0.0

    def register(self, builder):
        super().register(builder)
        self.environment = builder.particulator.environment

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

        self._download_to_buffer(self.particulator.environment["rhod"])
        rhod = self.buffer.copy()

        self._download_to_buffer(self.environment["z"])
        current_z = self.buffer
        result = cwc * rhod * (current_z - self.previous_z)
        self.previous_z = current_z
        return result


class ParcelLiquidWaterPath(ParcelCloudWaterPath):
    def __init__(self, unit="kg/m^2", name=None):
        super().__init__(unit=unit, name=name, liquid=True, ice=False)


class ParcelIceWaterPath(ParcelCloudWaterPath):
    def __init__(self, unit="kg/m^2", name=None):
        super().__init__(unit=unit, name=name, liquid=False, ice=True)
