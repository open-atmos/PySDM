"""
mean radius of particles within a grid cell (optionally restricted to a given size range)
"""

import numpy as np

from PySDM.products.impl import MomentProduct, register_product


@register_product()
class MeanRadius(MomentProduct):
    def __init__(
        self,
        name=None,
        unit="m",
        radius_range=(0, np.inf),
    ):
        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        builder.request_attribute("volume")
        super().register(builder)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="volume",
            rank=1 / 3,
            filter_range=(
                self.formulae.particle_shape_and_density.radius_to_mass(
                    self.radius_range[0]
                ),
                self.formulae.particle_shape_and_density.radius_to_mass(
                    self.radius_range[1]
                ),
            ),
            filter_attr="signed water mass",
        )
        self.buffer[:] /= self.formulae.constants.PI_4_3 ** (1 / 3)
        return self.buffer
