"""
Created at 11.05.2020
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import formulae as phys


class CriticalVolume(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        self.r_dry = builder.get_attribute('dry radius')
        self.environment = builder.core.environment
        self.particles = builder.core
        dependencies = [self.r_dry, self.cell_id]
        super().__init__(builder, name='critical volume', dependencies=dependencies)

    def recalculate(self):
        kappa = self.particles.dynamics['Condensation'].kappa
        r_d = self.r_dry.get().data
        T = self.environment['T'].data
        cell = self.cell_id.get().data
        for i in range(len(self.data)):  # TODO #347 move to backend
            self.data.data[i] = phys.volume(phys.r_cr(kp=kappa, rd=r_d[i], T=T[cell[i]]))
