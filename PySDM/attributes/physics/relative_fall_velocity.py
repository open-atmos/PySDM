"""
Attributes for tracking droplet velocity
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class RelativeFallMomentum(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="relative fall momentum", dtype=float)


class RelativeFallVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.momentum = builder.get_attribute("relative fall momentum")
        self.water_mass = builder.get_attribute("water mass")

        super().__init__(
            builder,
            name="relative fall velocity",
            dependencies=(self.momentum, self.water_mass),
        )

    def recalculate(self):
        self.data.ratio(self.momentum.get(), self.water_mass.get())
