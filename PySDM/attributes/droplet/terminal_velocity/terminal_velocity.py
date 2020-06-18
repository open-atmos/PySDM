"""
Created at 11.05.2020
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from .gunn_and_kinzer import RogersYau


class TerminalVelocity(DerivedAttribute):

    def __init__(self, particles_builder):
        self.radius = particles_builder.get_attribute('radius')
        dependencies = [self.radius]
        super().__init__(particles_builder, name='terminal velocity', dependencies=dependencies)

        self.approximation = RogersYau(particles_builder.particles)

    def recalculate(self):
        self.approximation(self.data, self.radius.get())

