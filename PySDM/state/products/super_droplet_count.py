"""
Created at 05.02.2020
"""

from PySDM.product import Product


class SuperDropletCount(Product):

    def __init__(self, particles_builder):
        super().__init__(
            core=particles_builder.core,
            shape=particles_builder.core.mesh.grid,
            name='n_sd',
            unit='#/gridbox',
            description='Super droplet count',
            scale='linear',
            range=[0, 10]
        )

    def get(self):
        cell_start = self.particles.state.cell_start
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            self.buffer.ravel()[i] = cell_start[i + 1] - cell_start[i]
        return self.buffer
