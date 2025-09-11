"""
Attributes for tracking particle velocity
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
        self.signed_water_mass = builder.get_attribute("signed water mass")

        super().__init__(
            builder,
            name="relative fall velocity",
            dependencies=(self.momentum, self.signed_water_mass),
        )

    def recalculate(self):
        self.data.ratio(self.momentum.get(), self.signed_water_mass.get())
