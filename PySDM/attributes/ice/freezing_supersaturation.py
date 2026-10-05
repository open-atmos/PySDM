"""
particle freezing supersaturation wrt ice
"""

from PySDM.attributes.impl import register_attribute
from ..impl import DerivedAttribute


@register_attribute()
class SupersaturationOfLastFreezing(DerivedAttribute):
    """time-dependent variant: assigned upon freezing"""

    def __init__(self, particulator):
        assert "Freezing" in particulator.dynamics
        assert particulator.dynamics["Freezing"].immersion_freezing != "singular"
        self.signed_water_mass = particulator.get_attribute("signed water mass")
        self.cell_id = particulator.get_attribute("cell id")
        super().__init__(
            particulator,
            name="supersaturation of last freezing",
            dependencies=(self.signed_water_mass, self.cell_id),
        )
        particulator.observers.append(self)

    def notify(self):
        self.update()

    def recalculate(self):
        self.particulator.backend.record_freezing_supersaturations(
            data=self.data,
            cell_id=self.cell_id.data,
            relative_humidity_ice=self.particulator.environment["RH_ice"],
            signed_water_mass=self.signed_water_mass.data,
        )
