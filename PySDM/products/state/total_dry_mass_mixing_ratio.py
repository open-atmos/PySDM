from ..product import MomentProduct
import numpy as np
from PySDM.physics.constants import convert_to, si


class TotalDryMassMixingRatio(MomentProduct):
    def __init__(self, density):
        super().__init__(
            name='q_dry',
            description='total dry mass mixing ratio',
            unit='μg/kg'
        )
        self.density = density

    def get(self):
        self.download_moment_to_buffer('dry volume', rank=1)
        self.buffer[:] *= self.density
        result = np.copy(self.buffer)
        self.download_moment_to_buffer('dry volume', rank=0)
        result[:] *= self.buffer
        self.download_to_buffer(self.particulator.environment['rhod'])
        result[:] /= self.particulator.mesh.dv
        result[:] /= self.buffer
        convert_to(result, si.ug / si.kg)
        return result
