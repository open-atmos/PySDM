"""
critical wet volume (kappa-Koehler, computed using actual temperature)
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class CriticalVolume(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute("cell id")
        self.v_dry = builder.get_attribute("dry volume")
        self.v_wet = builder.get_attribute("volume")
        self.kappa = builder.get_attribute("kappa")
        self.f_org = builder.get_attribute("dry volume organic fraction")
        self.environment = builder.particulator.environment
        self.particles = builder.particulator
        dependencies = [self.v_dry, self.v_wet, self.cell_id]
        super().__init__(builder, name="critical volume", dependencies=dependencies)

    def recalculate(self):
        self.particulator.backend.critical_volume(
            v_cr=self.data,
            kappa=self.kappa.get(),
            f_org=self.f_org.get(),
            v_dry=self.v_dry.get(),
            v_wet=self.v_wet.get(),
            T=self.environment["T"],
            cell=self.cell_id.get(),
        )


@register_attribute()
class WetToCriticalVolumeRatio(DerivedAttribute):
    def __init__(self, builder):
        self.critical_volume = builder.get_attribute("critical volume")
        self.volume = builder.get_attribute("volume")
        super().__init__(
            builder,
            name="wet to critical volume ratio",
            dependencies=(self.critical_volume, self.volume),
        )

    def recalculate(self):
        self.data.ratio(self.volume.get(), self.critical_volume.get())
