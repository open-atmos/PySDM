"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import MomentProduct
from PySDM.simulation.physics import constants as const
from PySDM.simulation.physics import formulae as phys


class AerosolConcentration(MomentProduct):
    def __init__(self, particles, radius_threshold):
        self.radius_threshold = radius_threshold

        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='n_a',
            unit='cm-3',
            description='Aerosol concentration',
            scale='log',
            range=[1e-1, 1e4]
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0,
                                       attr_range=[0, phys.volume(self.radius_threshold)])
        self.buffer[:] /= self.particles.mesh.dv
        const.convert_to(self.buffer, const.si.centimetre**-3)
        return self.buffer
