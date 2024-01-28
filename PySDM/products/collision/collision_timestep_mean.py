"""
Average collision timestep length used when adaptive timestepping is enabled in the
 `PySDM.dynamics.collisions.collision.Collision` dynamic (fetching a value reset the counter)
"""

import numba
import numpy as np

from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.products.impl.product import Product


class CollisionTimestepMean(Product):
    def __init__(self, unit="s", name=None):
        super().__init__(unit=unit, name=name)
        self.count = 0
        self.collision = None
        self.range = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.collision = self.particulator.dynamics["Collision"]
        self.range = self.collision.dt_coal_range

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def __get_impl(buffer, count, dt):
        buffer[:] = np.where(buffer[:] > 0, count * dt / buffer[:], np.nan)

    def _impl(self, **kwargs):
        self._download_to_buffer(self.collision.stats_n_substep)
        CollisionTimestepMean.__get_impl(self.buffer, self.count, self.particulator.dt)
        self.collision.stats_n_substep[:] = 0
        self.count = 0
        return self.buffer

    def notify(self):
        self.count += 1
