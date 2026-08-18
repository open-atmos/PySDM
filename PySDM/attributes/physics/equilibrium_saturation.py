"""
kappa-Koehler equilibrium saturation calculated for actual environment temperature
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class EquilibriumSaturation(DerivedAttribute):
    def __init__(self, particulator):
        self.r_wet = particulator.get_attribute("radius")
        self.v_wet = particulator.get_attribute("volume")
        self.v_dry = particulator.get_attribute("dry volume")
        self.kappa = particulator.get_attribute("kappa")
        self.f_org = particulator.get_attribute("dry volume organic fraction")

        super().__init__(
            particulator=particulator,
            name="equilibrium saturation",
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
