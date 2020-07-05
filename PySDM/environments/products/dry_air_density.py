"""
Created at 2020
"""

from ...product import Product
from PySDM.environments._moist import _Moist


class DryAirDensity(Product):

    def __init__(self, particles_builder):
        particles = particles_builder.particles
        assert isinstance(particles.environment, _Moist)
        self.environment = particles.environment
        super().__init__(particles=particles,
                         description="Dry-air density",
                         name="rhod",
                         unit="kg/m^3",
                         range=(0.95, 1.3),
                         scale="linear",
                         shape=particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['rhod'])
        return self.buffer

