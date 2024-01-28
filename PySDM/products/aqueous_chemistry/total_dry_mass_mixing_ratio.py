"""
dry aerosol mass summed over all particles in a grid box per mass of dry air
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class TotalDryMassMixingRatio(MomentProduct):
    def __init__(self, density, name=None, unit="kg/kg"):
        super().__init__(unit=unit, name=name)
        self.density = density

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(attr="dry volume", rank=1)
        self.buffer[:] *= self.density
        result = np.copy(self.buffer)
        self._download_moment_to_buffer(attr="dry volume", rank=0)
        result[:] *= self.buffer
        self._download_to_buffer(self.particulator.environment["rhod"])
        result[:] /= self.particulator.mesh.dv
        result[:] /= self.buffer
        return result
