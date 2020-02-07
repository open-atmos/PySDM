"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import MomentProduct
from PySDM.simulation.physics import constants as const
from PySDM.simulation.physics import formulae as phys


class ParticleMeanRadius(MomentProduct):
    def __init__(self, particles):
        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='radius_m1',
            unit='um',
            description='mean radius',
            scale='linear',
            range=[1, 50]
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=1, exponent=1/3)
        self.buffer[:] /= self.particles.mesh.dv
        self.buffer[:] *= phys.radius(volume=1)
        const.convert_to(self.buffer, const.si.micrometre)
        return self.buffer
