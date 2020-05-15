"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const


class DryRadius(DerivedAttribute):
    def __init__(self, particles_builder):
        self.volume_dry = particles_builder.get_attribute('dry volume')
        dependencies = [self.volume_dry]
        super().__init__(particles_builder, name='dry radius', dependencies=dependencies)

    def recalculate(self):
        self.particles.backend.multiply_out_of_place(self.data, self.volume_dry.get(), (3 / 4 / const.pi))
        self.particles.backend.power(self.data, (1 / 3))
