"""
Created at 05.02.2020
"""

from ...product import Product
from PySDM.environments._moist import _Moist


class DryAirPotentialTemperature(Product):

    def __init__(self, particles_builder):
        particles = particles_builder.core
        assert isinstance(particles.environment, _Moist)
        self.environment = particles.environment
        super().__init__(core=particles,
                         description="Dry-air potential temperature",
                         name="thd",
                         unit="K",
                         range=(275, 300),
                         scale="linear",
                         shape=particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['thd'])
        return self.buffer
