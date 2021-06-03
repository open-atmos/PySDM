from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const
import numpy as np


class CriticalVolume(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        self.v_dry = builder.get_attribute('dry volume')
        self.v_wet = builder.get_attribute('volume')
        self.environment = builder.core.environment
        self.particles = builder.core
        dependencies = [self.v_dry, self.v_wet, self.cell_id]
        super().__init__(builder, name='critical volume', dependencies=dependencies)

    def recalculate(self):
        self.core.bck.critical_volume(self.data,
            kappa=self.particles.dynamics['Condensation'].kappa,
            v_dry=self.v_dry.get(),
            v_wet=self.v_wet.get(),
            T=self.environment['T'],
            cell=self.cell_id.get()
        )
