from ...product import MomentProduct
from ....physics import constants as const
import numpy as np

class IceWaterContent(MomentProduct):

    def __init__(self, specific=True):
        super().__init__(
            name='qi',
            unit='g/kg' if specific else 'kg/m3',
            description=f'Ice water mixing ratio'
        )
        self.specific = specific

    def get(self):
        self.download_moment_to_buffer('volume', rank=0, filter_range=(-np.inf, 0))
        conc = self.buffer.copy()

        self.download_moment_to_buffer('volume', rank=1, filter_range=(-np.inf, 0))
        result = self.buffer.copy()
        result[:] *= -const.rho_i * conc  / self.particulator.mesh.dv

        if self.specific:
            self.download_to_buffer(self.particulator.environment['rhod'])
            result[:] /= self.buffer
            const.convert_to(result, const.si.gram / const.si.kilogram)
        return result