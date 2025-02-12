"""
particle mass attributes
in simulations involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""

from PySDM.attributes.impl import (
    ExtensiveAttribute,
    DerivedAttribute,
    register_attribute,
)


@register_attribute()
class SignedWaterMass(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="signed water mass")


@register_attribute(
    name="water mass",
    variant=lambda _, formulae: not formulae.particle_shape_and_density.supports_mixed_phase(),
)
class ViewWaterMass(DerivedAttribute):
    def __init__(self, builder):
        self.signed_water_mass = builder.get_attribute("signed water mass")

        super().__init__(
            builder,
            name="water mass",
            dependencies=(self.signed_water_mass,),
        )

    def mark_updated(self):
        self.signed_water_mass.mark_updated()

    def allocate(self, idx):
        pass

    def recalculate(self):
        pass

    def get(self):
        return self.signed_water_mass.data


@register_attribute(
    name="water mass",
    variant=lambda _, formulae: formulae.particle_shape_and_density.supports_mixed_phase(),
)
class AbsWaterMass(DerivedAttribute):
    def __init__(self, builder):
        self.signed_water_mass = builder.get_attribute("signed water mass")

        super().__init__(
            builder,
            name="water mass",
            dependencies=(self.signed_water_mass,),
        )

    def recalculate(self):
        self.data.fill(self.signed_water_mass.data)
        self.data.abs()
