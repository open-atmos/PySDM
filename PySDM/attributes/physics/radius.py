from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const


class Radius(DerivedAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute('volume')
        dependencies = [self.volume]
        super().__init__(builder, name='radius', dependencies=dependencies)

    def recalculate(self):
        self.data.idx = self.volume.data.idx
        self.data.product(self.volume.get(), 1/const.pi_4_3)
        self.data **= 1/3
