from ..product import MomentProduct
from ...physics import constants as const
import numpy as np


class WaterMixingRatio(MomentProduct):

    def __init__(self, radius_range, name='ql', description_prefix='liquid'):
        self.radius_range = radius_range
        self.volume_range = None
        super().__init__(
            name=name,
            unit='g/kg',
            description=f'{description_prefix} water mixing ratio'
        )

    def register(self, builder):
        super().register(builder)
        self.volume_range = self.formulae.trivia.volume(np.asarray(self.radius_range))
        self.radius_range = None

    def get(self):  # TODO #217
        self.download_moment_to_buffer('volume', rank=0, filter_range=self.volume_range, filter_attr='volume')
        conc = self.buffer.copy()

        self.download_moment_to_buffer('volume', rank=1, filter_range=self.volume_range, filter_attr='volume')
        result = self.buffer.copy()
        result[:] *= const.rho_w
        result[:] *= conc
        result[:] /= self.particulator.mesh.dv

        self.download_to_buffer(self.particulator.environment['rhod'])
        result[:] /= self.buffer
        const.convert_to(result, const.si.gram / const.si.kilogram)
        return result
