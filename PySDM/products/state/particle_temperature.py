"""
Created at 17.02.2020
"""

from PySDM.products.product import MomentProduct


class ParticleTemperature(MomentProduct):

    def __init__(self):
        super().__init__(
            name='T',
            unit='K',
            description='Particle temperature',
            scale='linear',
            range=[295, 305]
        )

    def register(self, builder):
        super().register(builder)
        builder.request_attribute('temperature')

    def get(self):
        self.download_moment_to_buffer(attr='temperature', rank=1)
        return self.buffer
