from PySDM.attributes.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const


class pH(DerivedAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute('volume')
        self.Hp = builder.get_attribute('Hp')
        dependencies = [self.volume, self.Hp]
        super().__init__(builder, name='pH', dependencies=dependencies)

    def recalculate(self):
        -ln_10(Hp / dv)
        self.data.idx = self.volume.data.idx
        self.data.product(self.volume.get(), (3 / 4 / const.pi))
        self.data **= 1 / 3
