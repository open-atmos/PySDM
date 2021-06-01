from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class ParticleMeanRadius(MomentProduct):

    def __init__(self):
        super().__init__(
            name='radius_m1',
            unit='um',
            description='Mean radius'
        )

    def get(self, unit=const.si.micrometre):
        self.download_moment_to_buffer('volume', rank=1/3)
        self.buffer[:] /= const.pi_4_3**(1/3)
        const.convert_to(self.buffer, unit)
        return self.buffer
