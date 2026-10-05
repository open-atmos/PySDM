"""
particle volume (derived from water mass);
in simulation involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class Volume(DerivedAttribute):
    def __init__(self, particulator):
        self.water_mass = particulator.get_attribute("water mass")
        super().__init__(particulator, name="volume", dependencies=(self.water_mass,))

    def recalculate(self):
        self.particulator.backend.volume_of_water_mass(self.data, self.water_mass.get())
