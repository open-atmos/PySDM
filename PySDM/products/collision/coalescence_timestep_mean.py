import numpy as np
import numba
from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.products.impl.product import Product


class CoalescenceTimestepMean(Product):

    def __init__(self, unit='s', name=None):
        super().__init__(unit=unit, name=name)
        self.count = 0
        self.coalescence = None
        self.range = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.coalescence = self.particulator.dynamics['Coalescence']
        self.range = self.coalescence.dt_coal_range

    @staticmethod
    @numba.njit(**JIT_FLAGS)
    def __get_impl(buffer, count, dt):
        buffer[:] = np.where(buffer[:] > 0, count * dt / buffer[:], np.nan)

    def _impl(self, **kwargs):
        self._download_to_buffer(self.coalescence.stats_n_substep)
        CoalescenceTimestepMean.__get_impl(self.buffer, self.count, self.particulator.dt)
        self.coalescence.stats_n_substep[:] = 0
        self.count = 0
        return self.buffer

    def notify(self):
        self.count += 1
