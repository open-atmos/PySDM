"""
particle volume (derived from water mass);
in simulation involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""

from PySDM.attributes.impl import DerivedAttribute


class Volume(DerivedAttribute):
    def __init__(self, builder):
        self.water_mass = builder.get_attribute("water mass")
        super().__init__(builder, name="volume", dependencies=(self.water_mass,))

    def recalculate(self):
        self.particulator.backend.volume_of_water_mass(self.data, self.water_mass.get())
