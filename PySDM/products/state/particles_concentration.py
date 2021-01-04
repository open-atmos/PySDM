"""
Created at 05.02.2020
"""

import numpy as np

from PySDM.physics import constants as const
from PySDM.physics import formulae as phys
from PySDM.products.product import MomentProduct


class ParticlesConcentration(MomentProduct):

    def __init__(self, radius_range, specific=False):
        self.radius_range = radius_range

        super().__init__(
            name='n_a_cm3',
            unit='mg-1' if specific else 'cm-3',
            description='Particles concentration',
            scale='linear',
            range=[1e0, 1e2]
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0,
                                       filter_range=(phys.volume(self.radius_range[0]),
                                                     phys.volume(self.radius_range[1])))
        self.buffer[:] /= self.core.mesh.dv
        const.convert_to(self.buffer, const.si.centimetre**-3)
        return self.buffer


class AerosolConcentration(ParticlesConcentration):

    def __init__(self, radius_threshold):
        super().__init__((0, radius_threshold))
        self.name = 'n_a_cm3'
        self.description = 'Aerosol particles concentration'


class CloudConcentration(ParticlesConcentration):

    def __init__(self, radius_range):
        super().__init__(radius_range)
        self.name = 'n_c_cm3'
        self.description = 'Cloud droplets concentration'


class DrizzleConcentration(ParticlesConcentration):

    def __init__(self, radius_threshold):
        super().__init__((radius_threshold, np.inf))
        self.name = 'n_d_cm3'
        self.description = 'Drizzle droplets concentration'
        self.scale = 'log'
        self.range = (1e-3, 1e1)
