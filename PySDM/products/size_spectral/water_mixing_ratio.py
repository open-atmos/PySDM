"""
liquid water mixing ratio (per mass of dry air) computed from particle sizes
 (optionally restricted to a given size range)
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class WaterMixingRatio(MomentProduct):
    def __init__(self, radius_range=None, name=None, unit="dimensionless"):
        self.radius_range = radius_range or (0, np.inf)
        self.volume_range = None
        super().__init__(unit=unit, name=name)

    def register(self, builder):
        super().register(builder)
        self.volume_range = self.formulae.trivia.volume(np.asarray(self.radius_range))
        self.radius_range = None

    def _impl(self, **kwargs):  # TODO #217
        self._download_moment_to_buffer(
            attr="volume", rank=0, filter_range=self.volume_range, filter_attr="volume"
        )
        conc = self.buffer.copy()

        self._download_moment_to_buffer(
            attr="volume", rank=1, filter_range=self.volume_range, filter_attr="volume"
        )
        result = self.buffer.copy()
        result[:] *= self.formulae.constants.rho_w
        result[:] *= conc
        result[:] /= self.particulator.mesh.dv

        self._download_to_buffer(self.particulator.environment["rhod"])
        result[:] /= self.buffer
        return result
