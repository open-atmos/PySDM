from ..product import MomentProduct
from ...physics import constants as const
from ...physics import formulae as phys
import numpy as np


class CloudWaterMixingRatio(MomentProduct):

    def __init__(self, radius_range):
        self.volume_range = phys.volume(np.asarray(radius_range))
        super().__init__(
            name='ql',
            unit='g/kg',
            description='cloud water mixing ratio',
            scale='linear',
            range=[0, 2]
        )

    def get(self):  # TODO #217
        self.download_moment_to_buffer('volume', rank=0, filter_range=self.volume_range, filter_attr='volume')
        conc = self.buffer.copy()

        self.download_moment_to_buffer('volume', rank=1, filter_range=self.volume_range, filter_attr='volume')
        result = self.buffer.copy()
        result[:] *= const.rho_w
        result[:] *= conc
        result[:] /= self.core.mesh.dv

        self.download_to_buffer(self.core.environment['rhod'])
        result[:] /= self.buffer
        const.convert_to(result, const.si.gram / const.si.kilogram)
        return result
