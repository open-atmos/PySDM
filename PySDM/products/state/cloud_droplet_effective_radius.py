import numpy as np
import numba
from ...backends.numba.conf import JIT_FLAGS
from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct

GEOM_FACTOR = const.pi_4_3**(-1/3)


class CloudDropletEffectiveRadius(MomentProduct):

    def __init__(self, radius_range):
        self.radius_range = radius_range

        super().__init__(
            name='r_eff',
            unit='um',
            description='Cloud Droplet Effective Radius'
        )

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def __get_impl(buffer, tmp):
        buffer[:] = np.where(tmp[:] > 0, buffer[:] * GEOM_FACTOR / tmp[:], np.nan)

    def get(self):
        tmp = np.empty_like(self.buffer)
        self.download_moment_to_buffer('volume', rank=2/3,
                                       filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                                                     self.formulae.trivia.volume(self.radius_range[1])))
        tmp[:] = self.buffer[:]
        self.download_moment_to_buffer('volume', rank=1,
                                       filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                                                     self.formulae.trivia.volume(self.radius_range[1])))
        CloudDropletEffectiveRadius.__get_impl(self.buffer, tmp)
        const.convert_to(self.buffer, const.si.micrometre)
        return self.buffer
