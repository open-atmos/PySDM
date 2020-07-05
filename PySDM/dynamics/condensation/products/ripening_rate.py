"""
Created at 05.02.2020
"""

from PySDM.product import Product
from PySDM.dynamics.condensation.condensation import Condensation


class RipeningRate(Product):

    def __init__(self, particles_builder):
        particles = particles_builder.core
        self.condensation = particles.dynamics[str(Condensation)]

        super().__init__(
            core=particles,
            shape=particles.mesh.grid,
            name='ripening_rate',
            description='ripening rate'
        )

    def get(self): # TODO: take into account NUMBER of substeps (?)
        self.download_to_buffer(self.condensation.ripening_flags)
        self.condensation.ripening_flags[:] = 0
        return self.buffer
