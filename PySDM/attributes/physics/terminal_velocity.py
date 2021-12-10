from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics.terminal_velocity.gunn_and_kinzer import Interpolation


class TerminalVelocity(DerivedAttribute):

    def __init__(self, builder):
        self.radius = builder.get_attribute('radius')
        dependencies = [self.radius]
        super().__init__(builder, name='terminal velocity', dependencies=dependencies)

        self.approximation = Interpolation(builder.particulator)

    def recalculate(self):
        self.approximation(self.data, self.radius.get())
