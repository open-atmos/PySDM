"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import Product
from PySDM.dynamics.condensation.condensation import Condensation
import numpy as np


class RipeningFlag(Product):
    def __init__(self, particles_builder):
        particles = particles_builder.particles
        self.condensation = particles.dynamics[str(Condensation)]

        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='ripening_flag',
            description='ripening flag'
        )



    def get(self):
        self.download_to_buffer(self.condensation.ripening_flags)
        self.particles.backend.fill(self.condensation.ripening_flags, 0)
        return self.buffer





