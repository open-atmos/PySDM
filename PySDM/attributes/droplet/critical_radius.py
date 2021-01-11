"""
Created at 11.05.2020
"""

from PySDM.attributes.derived_attribute import DerivedAttribute
from PySDM.physics import formulae as phys
from PySDM.dynamics import Condensation


class CriticalRadius(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        self.r_dry = builder.get_attribute('dry radius')
        self.environment = builder.core.environment
        self.particles = builder.core
        dependencies = [self.r_dry, self.cell_id]
        super().__init__(builder, name='critical radius', dependencies=dependencies)

    def recalculate(self):
        kappa = self.particles.dynamics['Condensation'].kappa
        r_d = self.r_dry.get()
        T = self.environment['T']
        cell = self.cell_id.get()
        for i in range(len(self.data)):  # TODO #347 move to backend
            self.data.data[i] = phys.r_cr(kp=kappa, rd=r_d[i], T=T[cell[i]])

