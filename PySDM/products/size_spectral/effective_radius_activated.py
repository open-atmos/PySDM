"""
effective radius of particles within a grid cell, for activated, unactivated or both
"""

import numba
import numpy as np

from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.products.impl.activation_filtered_product import _ActivationFilteredProduct
from PySDM.physics import constants as const
from PySDM.products.impl.moment_product import MomentProduct

GEOM_FACTOR = const.PI_4_3 ** (-1 / 3)


class ActivatedEffectiveRadius(MomentProduct, _ActivationFilteredProduct):
    def __init__(
        self,
        *,
        radius_range=None,
        count_unactivated: bool,
        count_activated: bool,
        name=None,
        unit="m"
    ):
        MomentProduct.__init__(self, name=name, unit=unit)
        _ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )
        self.volume_range = None
        self.radius_range = radius_range or (0, np.inf)

    def register(self, builder):
        _ActivationFilteredProduct.register(self, builder)
        MomentProduct.register(self, builder)
        self.volume_range = self.formulae.trivia.volume(np.asarray(self.radius_range))

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def __get_impl(buffer, tmp):
        buffer[:] = np.where(
            tmp[:] > 0,
            buffer[:]
            * GEOM_FACTOR
            / (tmp[:] + (tmp[:] == 0)),  # (+ x==0) to avoid div-by-zero warnings
            np.nan,
        )

    def _impl(self, **kwargs):
        _ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=2 / 3,
        )
        tmp = np.empty_like(self.buffer)
        tmp[:] = self.buffer[:]
        _ActivationFilteredProduct.impl(
            self,
            attr="volume",
            rank=1,
        )
        ActivatedEffectiveRadius.__get_impl(self.buffer, tmp)
        return self.buffer
