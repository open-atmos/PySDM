"""
Created at 17.02.2020
"""

from PySDM.product import MomentProduct


class ParticleTemperature(MomentProduct):

    def __init__(self, particles_builder):
        particles_builder.request_attribute('temperature')
        super().__init__(
            particles=particles_builder.particles,
            shape=particles_builder.particles.mesh.grid,
            name='T',
            unit='K',
            description='Particle temperature',
            scale='linear',
            range=[295, 305]
        )

    def get(self):
        self.download_moment_to_buffer(attr='temperature', rank=1)
        return self.buffer
