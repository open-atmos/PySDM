"""
particle dry volume (subject to evolution due to collisions or aqueous chemistry)
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class DryVolumeDynamic(DerivedAttribute):
    def __init__(self, builder):
        self.particulator = builder.particulator
        self.moles_sulphur_p6 = builder.get_attribute("moles_S_VI")
        super().__init__(
            builder, name="dry volume", dependencies=(self.moles_sulphur_p6,)
        )

    def recalculate(self):
        dynamic = self.particulator.dynamics["AqueousChemistry"]
        self.data.fill(self.moles_sulphur_p6.data)
        self.data *= dynamic.dry_molar_mass / dynamic.dry_rho


class DryVolume(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="dry volume")


class DryVolumeOrganic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="dry volume organic")


class OrganicFraction(DerivedAttribute):
    def __init__(self, builder):
        self.volume_dry_org = builder.get_attribute("dry volume organic")
        self.volume_dry = builder.get_attribute("dry volume")
        super().__init__(
            builder,
            name="dry volume organic fraction",
            dependencies=(self.volume_dry_org, self.volume_dry),
        )

    def recalculate(self):
        self.data.ratio(self.volume_dry_org.get(), self.volume_dry.get())
