"""
particle freezing temperature
"""

from PySDM.attributes.impl import MaximumAttribute, register_attribute
from ..impl import DerivedAttribute


@register_attribute()
class FreezingTemperature(MaximumAttribute):
    """singular variant: assigned at initialisation, modified through collisions only"""

    def __init__(self, builder):
        super().__init__(builder, name="freezing temperature")


@register_attribute()
class TemperatureOfLastFreezing(DerivedAttribute):
    """time-dependent variant: assigned upon freezing"""

    def __init__(self, builder):
        assert "Freezing" in builder.particulator.dynamics
        assert not builder.particulator.dynamics["Freezing"].singular
        self.signed_water_mass = builder.get_attribute("signed water mass")
        self.cell_id = builder.get_attribute("cell id")
        super().__init__(
            builder,
            name="temperature of last freezing",
            dependencies=(self.signed_water_mass, self.cell_id),
        )
        builder.particulator.observers.append(self)

    def notify(self):
        self.update()

    def recalculate(self):
        self.particulator.backend.record_freezing_temperatures(
            data=self.data,
            cell_id=self.cell_id.data,
            temperature=self.particulator.environment["T"],
            signed_water_mass=self.signed_water_mass.data,
        )
