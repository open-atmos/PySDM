"""
particle freezing temperature (assigned at initialisation, modified through collisions only,
 used in singular regime)
"""

from PySDM.attributes.impl import MaximumAttribute, register_attribute
from ..impl import DerivedAttribute


@register_attribute()
class FreezingTemperature(MaximumAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="freezing temperature")


@register_attribute()
class TemperatureOfLastFreezing(DerivedAttribute):
    def __init__(self, builder):
        self.signed_water_mass = builder.get_attribute("signed water mass")
        super().__init__(
            builder,
            name="temperature of last freezing",
            dependencies=(self.signed_water_mass,),
        )
