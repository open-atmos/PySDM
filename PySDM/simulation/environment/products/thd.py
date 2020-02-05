"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ...product import Product


class THD(Product):
    def __init__(self, environment):
        self.environment = environment
        super().__init__(particles=environment.particles,
                         description="Dry-air potential temperature",
                         name="thd",
                         unit="K",
                         range=(275, 300),
                         scale="linear",
                         shape=environment.particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['thd'])
        return self.buffer
