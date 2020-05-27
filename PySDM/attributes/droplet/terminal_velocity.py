"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from .radius import Radius
from PySDM.physics import constants as const


class TerminalVelocity(DerivedAttribute):

    def __init__(self, particles_builder):
        self.radius = particles_builder.get_attribute('radius')
        dependencies = [self.radius]
        super().__init__(particles_builder, name='terminal velocity', dependencies=dependencies)
        self.k1 = 1.19e6 / const.si.centimetre / const.si.second
        self.k2 = 8e3 / const.si.second
        self.k3 = 2.01e3 * const.si.centimetre ** (1 / 2) / const.si.second
        self.r1 = 40 * const.si.um
        self.r2 = 600 * const.si.um

    def recalculate(self):
        self.particles.backend.terminal_velocity(self.data, self.radius.get(), self.k1, self.k2, self.k3, self.r1, self.r2)
