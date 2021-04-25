"""
Created at 11.05.2020
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import formulae as phys
import numpy as np


class CriticalVolume(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        self.v_dry = builder.get_attribute('dry volume')
        self.v_wet = builder.get_attribute('volume')
        self.environment = builder.core.environment
        self.particles = builder.core
        self.formulae = builder.core.backend.formulae
        dependencies = [self.v_dry, self.v_wet, self.cell_id]
        super().__init__(builder, name='critical volume', dependencies=dependencies)

    def recalculate(self):
        kappa = self.particles.dynamics['Condensation'].kappa
        v_dry = self.v_dry.get().data
        v_wet = self.v_wet.get().data
        T = self.environment['T'].data
        cell = self.cell_id.get().data
        for i in range(len(self.data)):  # TODO #347 move to backend
            sigma = self.formulae.surface_tension.sigma(T[cell[i]], v_wet[i], v_dry[i])
            self.data.data[i] = phys.volume(self.formulae.hygroscopicity.r_cr(
                kp=kappa,
                rd3=v_dry[i] / (4/3*np.pi),
                T=T[cell[i]],
                sgm=sigma
            ))
