"""
kappa-Koehler hygroscopicity representation:
 the `PySDM.attributes.physics.hygroscopicity.KappaTimesDryVolume` base attribute
 and the derived `PySDM.attributes.physics.hygroscopicity.Kappa` attribute
"""

from ..impl import DerivedAttribute, ExtensiveAttribute, register_attribute


@register_attribute()
class KappaTimesDryVolume(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="kappa times dry volume")


@register_attribute()
class Kappa(DerivedAttribute):
    def __init__(self, builder):
        self.kappa_times_dry_volume = builder.get_attribute("kappa times dry volume")
        self.dry_volume = builder.get_attribute("dry volume")
        deps = (self.kappa_times_dry_volume, self.dry_volume)
        super().__init__(builder, name="kappa", dependencies=deps)

    def recalculate(self):
        self.data.ratio(self.kappa_times_dry_volume.get(), self.dry_volume.get())
