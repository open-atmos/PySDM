"""
kappa-Koehler equilibrium supersaturation calculated for actual environment temperature
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class EquilibriumSupersaturation(DerivedAttribute):
    def __init__(self, builder):
        self.r_wet = builder.get_attribute("radius")
        self.v_wet = builder.get_attribute("volume")
        self.v_dry = builder.get_attribute("dry volume")
        self.kappa = builder.get_attribute("kappa")
        self.f_org = builder.get_attribute("dry volume organic fraction")

        super().__init__(
            builder=builder,
            name="equilibrium supersaturation",
            dependencies=(self.kappa, self.v_dry, self.f_org, self.r_wet),
        )

    def recalculate(self):
        if len(self.particulator.environment["T"]) != 1:
            raise NotImplementedError()
        temperature = self.particulator.environment["T"][0]
        rd3 = self.v_dry.data.data / self.formulae.constants.PI_4_3
        sgm = self.formulae.surface_tension.sigma(
            temperature,
            self.v_wet.data.data,
            self.v_dry.data.data,
            self.f_org.data.data,
        )

        self.data.data[:] = self.formulae.hygroscopicity.RH_eq(
            self.r_wet.data.data,
            T=temperature,
            kp=self.kappa.data.data,
            rd3=rd3,
            sgm=sgm,
        )
