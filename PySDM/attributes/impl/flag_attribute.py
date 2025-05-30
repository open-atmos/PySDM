""" a Boolean attribute that gets reset for False after copying out the recorded values """

from .attribute import Attribute


class FlagAttribute(Attribute):
    def __init__(self, builder, name):
        super().__init__(builder, name, dtype=bool)
        builder.particulator.observers.append(self)

    def notify(self):
        source = self.particulator.dynamics["Collision"].flag_coalescence
        self.data[:] = source
        source[:] = False

    def allocate(self, idx):
        super().allocate(idx)
        self.data[:] = False

    def get(self):
        return self.data
