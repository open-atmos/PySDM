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
            range=[0, 5]
        )

    def get(self):  # TODO #217
        self.download_moment_to_buffer('volume', rank=0, filter_range=self.volume_range, filter_attr='volume')
        conc = self.buffer.copy()  # unit: #/cell

        self.download_moment_to_buffer('volume', rank=1, filter_range=self.volume_range, filter_attr='volume')
        result = self.buffer.copy()  # unit: m3
        result[:] *= const.rho_w  # unit: kg
        result[:] *= conc  # unit: kg / cell
        result[:] /= self.core.mesh.dv  #  unit: kg / m3

        self.download_to_buffer(self.core.environment['rhod'])
        result[:] /= self.buffer  # unit: kg / kg
        const.convert_to(result, const.si.gram / const.si.kilogram)
        return result
