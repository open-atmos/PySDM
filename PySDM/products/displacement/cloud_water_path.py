"""
cloud water path from integral over vertical domain
liquid water path, only liquid
ice water path, only ice
"""

import numpy as np

from PySDM.environments.parcel import Parcel

from PySDM.products.impl.moment_product import MomentProduct


class CloudWaterPath(MomentProduct):
    def __init__(self, name=None, unit="kg/m^2", liquid=True, ice=True):
        super().__init__(unit=unit, name=name)
        self.liquid = liquid
        self.ice = ice
        self.dz = None
        self.previous_z = None
        self.environment = None
        self._reset_counters()

    def _reset_counters(self):
        self.previous_z = 0.0

    def register(self, builder):
        super().register(builder)

        if isinstance(builder.particulator.environment, Parcel):
            self.dz = None
            self.environment = builder.particulator.environment
        else:
            self.dz = self.particulator.mesh.dz

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
            cwc += mass * conc / self.particulator.mesh.dv

        self._download_to_buffer(self.particulator.environment["rhod"])
        rhod = self.buffer.copy()

        if self.dz:
            result = cwc * rhod * self.dz
        else:
            self._download_to_buffer(self.environment["z"])
            current_z = self.buffer.copy()
            result = cwc * rhod * (current_z - self.previous_z)
            self.previous_z = current_z

        return result


class LiquidWaterPath(CloudWaterPath):
    def __init__(self, name=None, unit="kg/m^2"):
        super().__init__(name=name, unit=unit, liquid=True, ice=False)


class IceWaterPath(CloudWaterPath):
    def __init__(self, name=None, unit="kg/m^2"):
        super().__init__(name=name, unit=unit, liquid=False, ice=True)
