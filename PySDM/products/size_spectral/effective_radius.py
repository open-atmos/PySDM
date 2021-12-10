import numpy as np
import numba
from PySDM.physics import constants as const
from PySDM.products.impl.moment_product import MomentProduct
from PySDM.backends.impl_numba.conf import JIT_FLAGS

GEOM_FACTOR = const.PI_4_3 ** (-1 / 3)


class EffectiveRadius(MomentProduct):

    def __init__(self, *, radius_range=(0, np.inf), unit='m', name=None):
        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def __get_impl(buffer, tmp):
        buffer[:] = np.where(tmp[:] > 0, buffer[:] * GEOM_FACTOR / tmp[:], np.nan)

    def _impl(self, **kwargs):
        tmp = np.empty_like(self.buffer)
        self._download_moment_to_buffer(
            'volume', rank=2/3,
            filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                          self.formulae.trivia.volume(self.radius_range[1])))
        tmp[:] = self.buffer[:]
        self._download_moment_to_buffer(
            'volume', rank=1,
            filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                          self.formulae.trivia.volume(self.radius_range[1])))
        EffectiveRadius.__get_impl(self.buffer, tmp)
        return self.buffer
