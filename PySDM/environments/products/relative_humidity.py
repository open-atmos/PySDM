"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ...product import Product


class RelativeHumidity(Product):
    def __init__(self, environment):
        self.environment = environment
        super().__init__(particles=environment.particles,
                         description="Relative humidity",
                         name="RH",
                         unit="%",
                         range=(75, 105),
                         scale="linear",
                         shape=environment.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['RH'])
        self.buffer *= 100
        return self.buffer
