"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ...product import Product
from PySDM.simulation.physics import constants as const


class WaterVapourMixingRatio(Product):
    def __init__(self, environment):
        self.environment = environment
        super().__init__(particles=environment.particles,
                         description="Water vapour mixing ratio",
                         name="qv",
                         unit="g/kg",
                         range=(5, 7.5),
                         scale="linear",
                         shape=environment.particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['qv'])
        const.convert_to(self.buffer, const.si.gram / const.si.kilogram)
        return self.buffer
