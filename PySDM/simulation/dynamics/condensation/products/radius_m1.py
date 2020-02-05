"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import MomentProduct
from ....physics import constants as const
from ....physics import formulae as phys


class RadiusM1(MomentProduct):
    def __init__(self, condensation):
        particles = condensation.particles

        self.condensation = condensation

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
        self.buffer[:] /= const.si.micrometre
        return self.buffer
