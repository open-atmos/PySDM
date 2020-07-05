"""
Created at 05.02.2020
"""

from ...product import Product
from ...physics import constants as const
from PySDM.environments._moist import _Moist


class WaterVapourMixingRatio(Product):

    def __init__(self, particles_builder):
        particles = particles_builder.particles
        assert isinstance(particles.environment, _Moist)
        self.environment = particles.environment
        super().__init__(particles=particles,
                         description="Water vapour mixing ratio",
                         name="qv",
                         unit="g/kg",
                         range=(5, 7.5),
                         scale="linear",
                         shape=particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['qv'])
        const.convert_to(self.buffer, const.si.gram / const.si.kilogram)
        return self.buffer
