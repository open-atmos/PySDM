"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const


class Radius(DerivedAttribute):
    def __init__(self, particles_builder):
        self.volume = particles_builder.get_attribute('volume')
        dependencies = [self.volume]
        super().__init__(particles_builder, name='radius', dependencies=dependencies)

    def recalculate(self):
        self.particles.backend.multiply_out_of_place(self.data, self.volume.get(), (3 / 4 / const.pi))
        self.particles.backend.power(self.data, (1 / 3))
