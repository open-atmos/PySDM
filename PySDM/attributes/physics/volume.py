"""
particle (wet) volume, derived attribute for coalescence
in simulation involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""
from PySDM.attributes.impl import DerivedAttribute


class Volume(DerivedAttribute):
    def __init__(self, builder):
        self.water_mass = builder.get_attribute("water mass")
        super().__init__(builder, name="volume")

    def recalculate(self):
        self.data.product(self.water_mass.get(), 1 / self.formulae.constants.rho_w)
