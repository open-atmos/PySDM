"""
effective radius of particles within a grid cell (ratio of third to second moments,
 optionally restricted to a given size range)
"""

import numba
import numpy as np

from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.physics import constants as const
from PySDM.products.impl.moment_product import MomentProduct

GEOM_FACTOR = const.PI_4_3 ** (-1 / 3)


class EffectiveRadius(MomentProduct):
    def __init__(self, *, radius_range=None, unit="m", name=None):
        super().__init__(name=name, unit=unit)
        self.volume_range = None
        self.radius_range = radius_range or (0, np.inf)

    def register(self, builder):
        super().register(builder)
        self.volume_range = self.formulae.trivia.volume(np.asarray(self.radius_range))

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def nan_aware_reff_impl(input_volume_output_reff, volume_2_3):
        """computes the effective radius (<r^3>/<r^2>) based on <v^(2/3)> and <v>"""
        input_volume_output_reff[:] = np.where(
            volume_2_3[:] > 0,
            input_volume_output_reff[:]
            * GEOM_FACTOR
            / (
                volume_2_3[:] + (volume_2_3[:] == 0)
            ),  # (+ x==0) to avoid div-by-zero warnings
            np.nan,
        )

    def _impl(self, **kwargs):
        tmp = np.empty_like(self.buffer)
        self._download_moment_to_buffer(
            attr="volume",
            rank=2 / 3,
            filter_range=self.volume_range,
            filter_attr="volume",
        )
        tmp[:] = self.buffer[:]
        self._download_moment_to_buffer(
            attr="volume",
            rank=1,
            filter_range=self.volume_range,
            filter_attr="volume",
        )
        EffectiveRadius.nan_aware_reff_impl(
            input_volume_output_reff=self.buffer, volume_2_3=tmp
        )
        return self.buffer
