"""
Created at 05.02.2020
"""

from PySDM.physics import constants as const
from PySDM.physics import formulae as phys
from PySDM.products.product import MomentProduct


class ParticleMeanRadius(MomentProduct):

    def __init__(self):
        super().__init__(
            name='radius_m1',
            unit='um',
            description='Mean radius',
            scale='linear',
            range=[1, 25]
        )

    def get(self, unit=const.si.micrometre):
        self.download_moment_to_buffer('volume', rank=1/3)
        self.buffer[:] *= phys.radius(volume=1)
        const.convert_to(self.buffer, unit)
        return self.buffer
