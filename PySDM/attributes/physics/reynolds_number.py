"""
particle Reynolds number
"""

from ..impl.derived_attribute import DerivedAttribute


class ReynoldsNumber(DerivedAttribute):
    def __init__(self, builder):
        self.radius = builder.get_attribute("radius")
        self.velocity_wrt_air_motion = builder.get_attribute("relative fall velocity")
        super().__init__(
            builder,
            name="Reynolds number",
            dependencies=(self.radius, self.velocity_wrt_air_motion),
        )

    def recalculate(self):
        if len(self.particulator.environment["T"]) != 1:
            raise NotImplementedError()
        temperature = self.particulator.environment["T"][0]

        formulae = self.formulae
        air_dynamic_viscosity = formulae.air_dynamic_viscosity.eta_air(temperature)
        air_density = (
            formulae.state_variable_triplet.rho_of_rhod_and_water_vapour_mixing_ratio(
                rhod=self.particulator.environment["rhod"][0],
                water_vapour_mixing_ratio=self.particulator.environment[
                    "water vapour mixing ratio"
                ][0],
            )
        )

        self.data.data[:] = formulae.particle_shape_and_density.reynolds_number(
            air_dynamic_viscosity=air_dynamic_viscosity,
            air_density=air_density,
            radius=self.radius.data.data,
            velocity_wrt_air_motion=self.velocity_wrt_air_motion.data.data,
        )
