from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class TotalParticleSpecificConcentration(MomentProduct):

    def __init__(self):
        super().__init__(
            name='n_mg',
            unit='mg-1',
            description='Total particle specific concentration'
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0)
        result = self.buffer.copy()  # TODO #217
        self.download_to_buffer(self.core.environment['rhod'])
        result[:] /= self.core.mesh.dv
        result[:] /= self.buffer
        const.convert_to(result, const.si.milligram**-1)
        return result
