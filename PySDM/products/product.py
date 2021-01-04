"""
Created at 05.02.2020
"""

import numpy as np


class Product:

    def __init__(self, name, unit=None, description=None, scale=None, range=(0, 100)):
        self.name = name
        self.unit = unit
        self.description = description
        self.scale = scale
        self.range = range  # TODO: move out (maybe inject based on settings) and rename to something like plot_hint_range
        self.shape = None
        self.buffer = None
        self.core = None

    def register(self, builder):
        self.core = builder.core
        self.shape = self.core.mesh.grid
        self.buffer = np.empty(self.core.mesh.grid)

    def download_to_buffer(self, storage):
        storage.download(self.buffer.ravel())


class MomentProduct(Product):

    def __init__(self, name, unit, description, scale, range):
        super().__init__(name, unit, description, scale, range)
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.moments = self.core.Storage.empty((1, self.core.mesh.n_cell), dtype=float)

    def download_moment_to_buffer(self, attr, rank, filter_attr='volume', filter_range=(-np.inf, np.inf)):
        self.core.particles.moments(self.moment_0, self.moments, {attr: (rank,)}, attr_name=filter_attr, attr_range=filter_range)
        if rank == 0:  # TODO
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(self.moments.read_row(0))


