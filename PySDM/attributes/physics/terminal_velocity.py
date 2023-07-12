"""
particle terminal velocity (used for collision probability and particle displacement)
"""
from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.dynamics.terminal_velocity import Interpolation, RogersYau


class TerminalVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.radius = builder.get_attribute("radius")
        dependencies = [self.radius]
        super().__init__(builder, name="terminal velocity", dependencies=dependencies)

        # self.approximation = Interpolation(builder.particulator)
        self.approximation = RogersYau(particulator=builder.particulator)

    def recalculate(self):
        self.approximation(self.data, self.radius.get())
