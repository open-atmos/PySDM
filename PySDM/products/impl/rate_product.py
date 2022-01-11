"""
common code for products representing event rates
"""
from PySDM.products.impl.product import Product


class RateProduct(Product):
    def __init__(self, name, unit, counter, dynamic):
        super().__init__(name=name, unit=unit)
        self.timestep_count = 0
        self.counter = counter
        self.dynamic = dynamic

    def register(self, builder):
        super().register(builder)
        self.counter = getattr(self.particulator.dynamics[self.dynamic], self.counter)
        self.dynamic = None
        self.particulator.observers.append(self)

    def notify(self):
        self.timestep_count += 1

    def _impl(self, **kwargs):
        self._download_to_buffer(self.counter)
        if self.timestep_count != 0:
            self.buffer[:] /= self.timestep_count * self.particulator.dt
            self.timestep_count = 0
        self.counter[:] = 0
        return self.buffer
