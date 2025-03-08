"""
liquid water mixing ratio (per mass of dry air) computed from particle sizes
 (optionally restricted to a given size range)
"""

import numpy as np

from PySDM.products.impl import MomentProduct, register_product


@register_product()
class WaterMixingRatio(MomentProduct):
    def __init__(self, radius_range=None, name=None, unit="dimensionless"):
        self.radius_range = radius_range or (0, np.inf)
        self.signed_mass_range = None
        super().__init__(unit=unit, name=name)

    def register(self, builder):
        super().register(builder)
        self.signed_mass_range = (
            self.formulae.particle_shape_and_density.radius_to_mass(
                np.asarray(self.radius_range)
            )
        )
        self.radius_range = None

    def _impl(self, **kwargs):  # TODO #217
        self._download_moment_to_buffer(
            attr="water mass",
            rank=0,
            filter_range=self.signed_mass_range,
            filter_attr="signed water mass",
        )
        conc = self.buffer.copy()

        self._download_moment_to_buffer(
            attr="water mass",
            rank=1,
            filter_range=self.signed_mass_range,
            filter_attr="signed water mass",
        )
        result = self.buffer.copy()
        result[:] *= conc
        result[:] /= self.particulator.mesh.dv

        self._download_to_buffer(self.particulator.environment["rhod"])
        result[:] /= self.buffer
        return result
