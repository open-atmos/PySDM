"""
particle terminal velocity (used for collision probability and particle displacement)
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute(
    name="relative fall velocity",
    variant=lambda dynamics, _: "RelaxedVelocity" not in dynamics,
)
@register_attribute()
class TerminalVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.radius = builder.get_attribute("radius")
        self.signed_water_mass = builder.get_attribute("signed water mass")
        dependencies = [self.radius,self.signed_water_mass]
        super().__init__(builder, name="terminal velocity", dependencies=dependencies)

        self.approximation_liquid = builder.formulae.terminal_velocity_class(
            builder.particulator
        )
        self.approximation_ice = builder.formulae.terminal_velocity_ice_class(
            builder.particulator
        )

    def recalculate(self):
        print( "recalculate terminal_velocity" )
        self.approximation_liquid(self.data, self.radius.get())
