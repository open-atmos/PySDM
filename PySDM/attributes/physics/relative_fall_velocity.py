"""
Attributes for tracking droplet velocity
"""

from PySDM.attributes.impl import (
    DerivedAttribute,
    ExtensiveAttribute,
    register_attribute,
)


@register_attribute(
    name="relative fall momentum",
    variant=lambda dynamics, _: "RelaxedVelocity" in dynamics,
    dummy_default=True,
    warn=True,
    # note: could eventually make an attribute that calculates momentum
    #       from terminal velocity instead when no RelaxedVelocity dynamic is present
)
class RelativeFallMomentum(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="relative fall momentum", dtype=float)


@register_attribute(
    name="relative fall velocity",
    variant=lambda dynamics, _: "RelaxedVelocity" in dynamics,
)
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
