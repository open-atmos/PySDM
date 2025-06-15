"""
kappa-Koehler critical saturation calculated for either initial or actual environment temperature
"""

from PySDM.attributes.impl import (
    DerivedAttribute,
    register_attribute,
    TemperatureVariationOptionAttribute,
)


@register_attribute()
class CriticalSaturation(DerivedAttribute, TemperatureVariationOptionAttribute):
    def __init__(self, builder, neglect_temperature_variations=False):
        assert builder.particulator.mesh.dimension == 0

        self.v_crit = builder.get_attribute("critical volume")
        self.v_dry = builder.get_attribute("dry volume")
        self.kappa = builder.get_attribute("kappa")
        self.f_org = builder.get_attribute("dry volume organic fraction")
        TemperatureVariationOptionAttribute.__init__(
            self, builder, neglect_temperature_variations
        )
        DerivedAttribute.__init__(
            self,
            builder=builder,
            name="critical saturation",
            dependencies=(self.v_crit, self.kappa, self.v_dry, self.f_org),
        )

    def recalculate(self):
        temperature = (
            self.initial_temperature
            if self.neglect_temperature_variations
            else self.particulator.environment["T"]
        )
        r_cr = self.formulae.trivia.radius(self.v_crit.data.data)
        rd3 = self.v_dry.data.data / self.formulae.constants.PI_4_3
        sgm = self.formulae.surface_tension.sigma(
            temperature.data,
            self.v_crit.data.data,
            self.v_dry.data.data,
            self.f_org.data.data,
        )

        self.data.data[:] = self.formulae.hygroscopicity.RH_eq(
            r_cr, T=temperature.data, kp=self.kappa.data.data, rd3=rd3, sgm=sgm
        )


@register_attribute()
class CriticalSaturationNeglectingTemperatureVariations(CriticalSaturation):
    def __init__(self, builder):
        super().__init__(builder, neglect_temperature_variations=True)
