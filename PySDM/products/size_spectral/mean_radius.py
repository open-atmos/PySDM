"""
mean radius of particles within a grid cell (optionally restricted to a given size range)
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class MeanRadius(MomentProduct):
    def __init__(
        self,
        name=None,
        unit="m",
        radius_range=(0, np.inf),
    ):
        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="volume",
            rank=1 / 3,
            filter_range=(
                self.formulae.trivia.volume(self.radius_range[0]),
                self.formulae.trivia.volume(self.radius_range[1]),
            ),
            filter_attr="volume",
        )
        self.buffer[:] /= self.formulae.constants.PI_4_3 ** (1 / 3)
        return self.buffer
