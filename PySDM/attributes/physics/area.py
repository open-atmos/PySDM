"""
particle wet radius (calculated from the volume)
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class Area(DerivedAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute("volume")
        dependencies = [self.volume]
        super().__init__(builder, name="area", dependencies=dependencies)

    def recalculate(self):
        self.data.product(self.volume.get(), 1 / self.formulae.constants.PI_4_3)
        self.data **= 2 / 3
        self.data *= self.formulae.constants.PI_4_3 * 3
