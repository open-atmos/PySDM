from ..impl.derived_attribute import DerivedAttribute
from ..impl.extensive_attribute import ExtensiveAttribute


class KappaTimesDryVolume(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='kappa times dry volume')


class Kappa(DerivedAttribute):
    def __init__(self, builder):
        self.kappa_times_dry_volume = builder.get_attribute('kappa times dry volume')
        self.dry_volume = builder.get_attribute('dry volume')
        deps = (self.kappa_times_dry_volume, self.dry_volume)
        super().__init__(builder, name='kappa', dependencies=deps)

    def recalculate(self):
        self.data.ratio(self.kappa_times_dry_volume.get(), self.dry_volume.get())
