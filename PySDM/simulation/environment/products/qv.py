"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ...product import Product


class QV(Product):
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
        return self.buffer
