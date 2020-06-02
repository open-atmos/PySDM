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
        self.data.product(self.volume_dry.get(), (3 / 4 / const.pi))
        self.data **= 1 / 3
