"""
Created at 05.02.2020
"""

from PySDM.product import MomentProduct
from PySDM.physics import constants as const
from PySDM.physics import formulae as phys


class ParticleMeanRadius(MomentProduct):

    def __init__(self, particles_builder):
        super().__init__(
            particles=particles_builder.particles,
            shape=particles_builder.particles.mesh.grid,
            name='radius_m1',
            unit='um',
            description='mean radius',
            scale='linear',
            range=[1, 50]
        )

    def get(self, unit=const.si.micrometre):
        self.download_moment_to_buffer('volume', rank=1/3)
        self.buffer[:] *= phys.radius(volume=1)
        const.convert_to(self.buffer, unit)
        return self.buffer
