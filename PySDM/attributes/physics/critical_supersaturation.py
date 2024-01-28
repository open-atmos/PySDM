"""
kappa-Koehler critical supersaturation calculated for actual environment temperature
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class CriticalSupersaturation(DerivedAttribute):
    def __init__(self, builder):
        self.v_crit = builder.get_attribute("critical volume")
        self.v_dry = builder.get_attribute("dry volume")
        self.kappa = builder.get_attribute("kappa")
        self.f_org = builder.get_attribute("dry volume organic fraction")

        super().__init__(
            builder=builder,
            name="critical supersaturation",
            dependencies=(self.v_crit, self.kappa, self.v_dry, self.f_org),
        )

    def recalculate(self):
        if len(self.particulator.environment["T"]) != 1:
            raise NotImplementedError()
        temperature = self.particulator.environment["T"][0]
        r_cr = self.formulae.trivia.radius(self.v_crit.data.data)
        rd3 = self.v_dry.data.data / self.formulae.constants.PI_4_3
        sgm = self.formulae.surface_tension.sigma(
            temperature,
            self.v_crit.data.data,
            self.v_dry.data.data,
            self.f_org.data.data,
        )

        self.data.data[:] = self.formulae.hygroscopicity.RH_eq(
            r_cr, T=temperature, kp=self.kappa.data.data, rd3=rd3, sgm=sgm
        )
