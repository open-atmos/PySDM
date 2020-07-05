"""
Created at 05.02.2020
"""

import numpy as np


class Product:

    def __init__(self, core, shape, name, unit=None, description=None, scale=None, range=(0, 100)):
        self.name = name
        self.unit = unit
        self.description = description
        self.scale = scale
        self.range = range  # TODO: move out (maybe inject based on setup) and rename to something like plot_hint_range
        self.shape = shape
        self.buffer = np.empty(core.mesh.grid)
        self.particles = core

    def download_to_buffer(self, storage):
        storage.download(self.buffer.ravel())


class MomentProduct(Product):

    def __init__(self, core, shape, name, unit, description, scale, range):
        super().__init__(core, shape, name, unit, description, scale, range)
        self.particles = core
        # TODO
        self.moment_0 = core.backend.Storage.empty(core.mesh.n_cell, dtype=int)
        self.moments = core.backend.Storage.empty((1, core.mesh.n_cell), dtype=float)

    def download_moment_to_buffer(self, attr, rank, attr_range=(-np.inf, np.inf)):
        self.particles.state.moments(self.moment_0, self.moments, {attr: (rank,)}, attr_range=attr_range)
        if rank == 0:  # TODO
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(self.moments.read_row(0))


