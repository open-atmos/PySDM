"""
particle dry volume (subject to evolution due to collisions or aqueous chemistry)
"""

from PySDM.attributes.impl import (
    DerivedAttribute,
    ExtensiveAttribute,
    register_attribute,
)


@register_attribute(
    name="dry volume", variant=lambda dynamics, _: "AqueousChemistry" in dynamics
)
class DryVolumeDynamic(DerivedAttribute):
    def __init__(self, particulator):
        self.particulator = particulator
        self.moles_sulphur_p6 = particulator.get_attribute("moles_S_VI")
        super().__init__(
            particulator, name="dry volume", dependencies=(self.moles_sulphur_p6,)
        )

    def recalculate(self):
        dynamic = self.particulator.dynamics["AqueousChemistry"]
        self.data.fill(self.moles_sulphur_p6.data)
        self.data *= dynamic.dry_molar_mass / dynamic.dry_rho


@register_attribute(
    name="dry volume", variant=lambda dynamics, _: "AqueousChemistry" not in dynamics
)
class DryVolume(ExtensiveAttribute):
    def __init__(self, particulator):
        super().__init__(particulator, name="dry volume")


@register_attribute(
    name="dry volume organic",
    variant=lambda _, formulae: formulae.surface_tension.__name__ != "Constant",
    dummy_default=True,
)
class DryVolumeOrganic(ExtensiveAttribute):
    def __init__(self, particulator):
        super().__init__(particulator, name="dry volume organic")


@register_attribute(
    name="dry volume organic fraction",
    variant=lambda _, formulae: formulae.surface_tension.__name__ != "Constant",
    dummy_default=True,
)
class OrganicFraction(DerivedAttribute):
    def __init__(self, particulator):
        self.volume_dry_org = particulator.get_attribute("dry volume organic")
        self.volume_dry = particulator.get_attribute("dry volume")
        super().__init__(
            particulator,
            name="dry volume organic fraction",
            dependencies=(self.volume_dry_org, self.volume_dry),
        )

    def recalculate(self):
        self.data.ratio(self.volume_dry_org.get(), self.volume_dry.get())
