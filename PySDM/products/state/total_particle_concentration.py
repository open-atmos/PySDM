from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class TotalParticleConcentration(MomentProduct):

    def __init__(self):
        super().__init__(
            name='n_cm3',
            unit='cm-3',
            description='Total particle concentration'
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0)
        self.buffer[:] /= self.core.mesh.dv
        const.convert_to(self.buffer, const.si.centimetre**-3)
        return self.buffer
