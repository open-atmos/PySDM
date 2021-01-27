"""
Created at 23.11.2020
"""

import numpy as np

from PySDM.products.product import Product


class CoalescenceTimestep(Product):

    def __init__(self, debug=False):
        super().__init__(
            name='dt_coal',
            unit='s',
            description='Coalescence timestep',
            scale='log',
            range=None
        )
        self.minimum = None
        self.maximum = None
        self.count = None
        self.coalescence = None

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)
        self.coalescence = self.core.dynamics['Coalescence']
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
        self.download_to_buffer(self.coalescence.n_substep)
        self.buffer[:] = self.coalescence.core.dt / self.buffer
        return self.buffer

    def notify(self):
        self.download_to_buffer(self.coalescence.n_substep)
        self.count[:] += self.buffer
        self.buffer[:] = self.coalescence.core.dt / self.buffer
        self.minimum = np.minimum(self.buffer, self.minimum)
        self.maximum = np.maximum(self.buffer, self.maximum)

    def reset(self):
        self.minimum[:] = np.inf
        self.maximum[:] = -np.inf
        self.count[:] = 0
