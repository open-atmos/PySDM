"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import MomentProduct
from PySDM.simulation.physics import constants as const


class TotalParticleConcentration(MomentProduct):
    def __init__(self, particles):
        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='n_cm3',
            unit='cm-3',
            description='Total particle concentration',
            scale='linear',
            range=[20, 50]
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0)
        self.buffer[:] /= self.particles.mesh.dv
        const.convert_to(self.buffer, const.si.centimetre**-3)
        return self.buffer
