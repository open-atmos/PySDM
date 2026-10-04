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
    def __init__(self, particulator):
        self.radius = particulator.get_attribute("radius")
        self.signed_water_mass = particulator.get_attribute("signed water mass")
        self.cell_id = particulator.get_attribute("cell id")
        dependencies = [self.radius, self.signed_water_mass, self.cell_id]
        super().__init__(
            particulator, name="terminal velocity", dependencies=dependencies
        )

        self.approximation_liquid = particulator.formulae.terminal_velocity_class(
            particulator
        )
        self.approximation_ice = particulator.formulae.terminal_velocity_ice_class(
            particulator
        )

    def recalculate(self):
        self.approximation_liquid(self.data, self.radius.get())
        # TODO #1605 order of functions calls changes result. approximation_liquid will override
        #  approximation_ice when mixed-phase spheres shape active
        if self.formulae.particle_shape_and_density.supports_mixed_phase():
            self.approximation_ice(
                self.data,
                self.signed_water_mass.get(),
                self.cell_id.get(),
                self.particulator.environment["T"],
                self.particulator.environment["p"],
            )
