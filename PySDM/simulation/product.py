"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Product:
    def __init__(self, particles, shape, name, unit, description, scale, range):
        self.name = name
        self.unit = unit
        self.description = description
        self.scale = scale
        self.range = range  # TODO: rename to something like plot_hint_range

        self.buffer = np.empty(shape)
        self.backend = particles.backend
        self.particles = particles

    def download_to_buffer(self, storage):
        self.backend.download(storage, self.buffer.ravel())


class MomentProduct(Product):
    def __init__(self, particles, shape, name, unit, description, scale, range):
        super().__init__(particles, shape, name, unit, description, scale, range)

        # TODO
        self.moment_0 = particles.backend.array(particles.mesh.n_cell, dtype=int)
        self.moments = particles.backend.array((1, particles.mesh.n_cell), dtype=float)

    def download_moment_to_buffer(self, attr, rank, attr_range=(-np.inf, np.inf)):
        self.particles.state.moments(self.moment_0, self.moments, {attr: (rank,)}, attr_range=attr_range)
        if rank == 0:  # TODO
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(self.moments[0])

