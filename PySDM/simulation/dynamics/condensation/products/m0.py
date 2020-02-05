"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import MomentProduct
from ....physics import constants as const
from ....physics import formulae as phys


class M0(MomentProduct):
    def __init__(self, condensation):
        particles = condensation.particles

        self.condensation = condensation

        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='m0',
            unit='TODO',
            description='m0',
            scale='linear',
            range=[0, 1e8]
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0, exponent=1)  # TODO
        self.buffer[:] /= self.particles.mesh.dv
        return self.buffer
