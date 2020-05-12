"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .attribute import Attribute


class BaseAttribute(Attribute):

    def __init__(self, particles_builder, name, dtype=float):
        super().__init__(particles_builder, name, dtype)

    def init(self, data=None):
        if self.data is None:
            self.allocate()
        self.particles.backend.upload(data, self.data)
        self.update()
