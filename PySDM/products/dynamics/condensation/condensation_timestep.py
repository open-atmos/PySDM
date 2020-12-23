"""
Created at 05.02.2020
"""

import numpy as np

from PySDM.products.product import Product


class CondensationTimestep(Product):

    def __init__(self, debug=False):
        super().__init__(
            name='dt_cond',
            unit='s',
            description='Condensation timestep',
            scale='log',
            range=None
        )
        self.minimum = None
        self.maximum = None
        self.count = None
        self.condensation = None

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)
        self.condensation = self.core.dynamics['Condensation']
        self.range = (1e-5, self.core.dt)
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
        self.buffer[:] = self.condensation.core.dt / self.buffer
        return self.buffer

    def notify(self):
        self.download_to_buffer(self.condensation.substeps)
        self.count[:] += self.buffer
        self.buffer[:] = self.condensation.core.dt / self.buffer
        self.minimum = np.minimum(self.buffer, self.minimum)
        self.maximum = np.maximum(self.buffer, self.maximum)

    def reset(self):
        self.minimum[:] = np.inf
        self.maximum[:] = -np.inf
        self.count[:] = 0
