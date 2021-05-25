from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute
from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const


class DryVolumeStatic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume')


class DryVolumeDynamic(DerivedAttribute):
    def __init__(self, builder):
        self.core = builder.core
        self.moles_S_VI = builder.get_attribute('moles_S_VI')
        super().__init__(builder, name='dry volume', dependencies=(self.moles_S_VI,))

    def recalculate(self):
        dynamic = self.core.dynamics['AqueousChemistry']
        self.data.data[:] = self.moles_S_VI.data.data[:]
        self.data.data[:] *= dynamic.dry_molar_mass / dynamic.dry_rho
