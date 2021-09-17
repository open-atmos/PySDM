from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute
from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class DryVolumeDynamic(DerivedAttribute):
    def __init__(self, builder):
        self.particulator = builder.particulator
        self.moles_S_VI = builder.get_attribute('moles_S_VI')
        super().__init__(builder, name='dry volume', dependencies=(self.moles_S_VI,))

    def recalculate(self):
        dynamic = self.particulator.dynamics['AqueousChemistry']
        self.data.data[:] = self.moles_S_VI.data.data[:]
        self.data.data[:] *= dynamic.dry_molar_mass / dynamic.dry_rho


class DryVolume(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume')


class DryVolumeOrganic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume organic')


class OrganicFraction(DerivedAttribute):
    def __init__(self, builder):
        self.volume_dry_org = builder.get_attribute('dry volume organic')
        self.volume_dry = builder.get_attribute('dry volume')
        dependencies = [self.volume_dry_org, self.volume_dry]
        super().__init__(builder, name='dry volume organic fraction', dependencies=dependencies)

    def recalculate(self):
        self.data.ratio(self.volume_dry_org.get(), self.volume_dry.get())
