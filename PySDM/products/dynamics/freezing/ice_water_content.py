from ...product import MomentProduct
from ....physics import constants as const
import numpy as np

class IceWaterContent(MomentProduct):

    def __init__(self):
        super().__init__(
            name='qi',
            unit='g/kg',
            description=f'Ice water mixing ratio'
        )

    def get(self):
        self.download_moment_to_buffer('spheroid mass', rank=0)
        conc = self.buffer.copy()

        self.download_moment_to_buffer('spheroid mass', rank=1)
        result = self.buffer.copy()
        result[:] *= conc
        result[:] /= self.core.mesh.dv

        self.download_to_buffer(self.core.environment['rhod'])
        result[:] /= self.buffer
        const.convert_to(result, const.si.gram / const.si.kilogram)
        return result