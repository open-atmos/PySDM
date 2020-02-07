"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.product import Product


class CondensationTimestep(Product):
    def __init__(self, condensation):
        particles = condensation.particles

        self.condensation = condensation

        super().__init__(
            particles=particles,
            shape=particles.mesh.grid,
            name='dt_cond',
            unit='s',
            description='condensation timestep',
            scale='log',
            range=[1e-5, particles.dt]
        )

    def get(self):
        self.download_to_buffer(self.condensation.substeps)
        self.buffer[:] = self.condensation.particles.dt / self.buffer
        return self.buffer
