"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Product:
    def __init__(self, particles, shape, name, unit = None, description = None, scale = None, range = [0, 100]):
        self.name = name
        self.unit = unit
        self.description = description
        self.scale = scale
        self.range = range  # TODO: move out (maybe inject based on setup) and rename to something like plot_hint_range

        self.buffer = np.empty(shape)
        self.particles = particles

    def download_to_buffer(self, storage):
        self.particles.backend.download(storage, self.buffer.ravel())

    def poll(self):
        pass


class MomentProduct(Product):
    def __init__(self, particles, shape, name, unit, description, scale, range):
        super().__init__(particles, shape, name, unit, description, scale, range)
        self.particles = particles
        # TODO
        self.moment_0 = particles.backend.array(particles.mesh.n_cell, dtype=int)
        self.moments = particles.backend.array((1, particles.mesh.n_cell), dtype=float)

    def download_moment_to_buffer(self, attr, rank, attr_range=(-np.inf, np.inf)):
        self.particles.state.moments(self.moment_0, self.moments, {attr: (rank,)}, attr_range=attr_range)
        if rank == 0:  # TODO
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(self.particles.backend.read_row(self.moments, 0))


