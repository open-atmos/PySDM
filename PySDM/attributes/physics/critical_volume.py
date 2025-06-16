"""
critical wet volume (kappa-Koehler, computed using actual or initial temperature)
"""

from PySDM.attributes.impl import (
    DerivedAttribute,
    register_attribute,
    TemperatureVariationOptionAttribute,
)


@register_attribute()
class CriticalVolume(DerivedAttribute, TemperatureVariationOptionAttribute):
    def __init__(self, builder, neglect_temperature_variations=False):
        self.cell_id = builder.get_attribute("cell id")
        self.v_dry = builder.get_attribute("dry volume")
        self.v_wet = builder.get_attribute("volume")
        self.kappa = builder.get_attribute("kappa")
        self.f_org = builder.get_attribute("dry volume organic fraction")
        self.environment = builder.particulator.environment
        self.particles = builder.particulator

        dependencies = [self.v_dry, self.v_wet, self.cell_id]
        TemperatureVariationOptionAttribute.__init__(
            self, builder, neglect_temperature_variations
        )
        DerivedAttribute.__init__(
            self, builder, name="critical volume", dependencies=dependencies
        )

    def recalculate(self):
        temperature = (
            self.initial_temperature
            if self.neglect_temperature_variations
            else self.environment["T"]
        )
        self.particulator.backend.critical_volume(
            v_cr=self.data,
            kappa=self.kappa.get(),
            f_org=self.f_org.get(),
            v_dry=self.v_dry.get(),
            v_wet=self.v_wet.get(),
            T=temperature,
            cell=self.cell_id.get(),
        )


@register_attribute()
class CriticalVolumeNeglectingTemperatureVariations(CriticalVolume):
    def __init__(self, builder):
        super().__init__(builder, neglect_temperature_variations=True)


@register_attribute()
class WetToCriticalVolumeRatio(DerivedAttribute):
    def __init__(
        self,
        builder,
        neglect_temperature_variations=False,
        name="wet to critical volume ratio",
    ):
        self.critical_volume = builder.get_attribute(
            "critical volume"
            + (
                " neglecting temperature variations"
                if neglect_temperature_variations
                else ""
            )
        )
        self.volume = builder.get_attribute("volume")
        super().__init__(
            builder,
            name=name,
            dependencies=(self.critical_volume, self.volume),
        )

    def recalculate(self):
        self.data.ratio(self.volume.get(), self.critical_volume.get())


@register_attribute()
class WetToCriticalVolumeRatioNeglectingTemperatureVariations(WetToCriticalVolumeRatio):
    def __init__(self, builder):
        super().__init__(
            builder,
            neglect_temperature_variations=True,
            name="wet to critical volume ratio neglecting temperature variations",
        )
