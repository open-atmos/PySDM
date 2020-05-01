"""
Created at 17.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import MomentProduct


class ParticleTemperature(MomentProduct):
    def __init__(self, particles):
        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='T',
            unit='K',
            description='Particle temperature',
            scale='linear',
            range=[295, 305]
        )

    def get(self):
        self.download_moment_to_buffer(attr='temperature', rank=1)
        return self.buffer
