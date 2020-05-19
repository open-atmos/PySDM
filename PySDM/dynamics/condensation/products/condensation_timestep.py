"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import Product
from PySDM.dynamics.condensation.condensation import Condensation
import numpy as np


class CondensationTimestep(Product):
    def __init__(self, particles_builder):
        particles = particles_builder.particles
        self.condensation = particles.dynamics[str(Condensation)]

        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='dt_cond',
            unit='s',
            description='condensation timestep',
            scale='log',
            range=[1e-5, particles.dt]
        )
        self.minimum = np.full_like(self.buffer, np.nan)
        self.maximum = np.full_like(self.buffer, np.nan)
        self.count = np.full_like(self.buffer, np.nan)

    def get_min(self):
        return self.minimum

    def get_max(self):
        return self.maximum

    def get_count(self):
        return self.count

    def get(self):
        self.download_to_buffer(self.condensation.substeps)
        self.buffer[:] = self.condensation.particles.dt / self.buffer
        return self.buffer


    def poll(self):
        if self.debug:
            self.download_to_buffer(self.condensation.substeps)
            self.count[:] += self.buffer
            self.buffer[:] = self.condensation.particles.dt / self.buffer
            self.minimum = np.minimum(self.buffer, self.minimum)
            self.maximum = np.maximum(self.buffer, self.maximum)

    def reset(self):
        self.minimum[:] = np.inf
        self.maximum[:] = -np.inf
        self.count[:] = 0




