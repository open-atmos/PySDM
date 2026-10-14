"""
particle wet radius (calculated from the volume)
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class Area(DerivedAttribute):
    def __init__(self, particulator):
        self.volume = particulator.get_attribute("volume")
        dependencies = [self.volume]
        super().__init__(particulator, name="area", dependencies=dependencies)

    def recalculate(self):
        self.data.product(self.volume.get(), 1 / self.formulae.constants.PI_4_3)
        self.data **= 2 / 3
        self.data *= self.formulae.constants.PI_4_3 * 3
