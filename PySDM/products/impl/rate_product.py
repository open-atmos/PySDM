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
        result = self.buffer.copy()
        if self.timestep_count != 0:
            result[:] /= (
                self.timestep_count * self.particulator.dt * self.particulator.mesh.dv
            )
            self.timestep_count = 0
            self._download_to_buffer(self.particulator.environment["rhod"])
            result[:] /= self.buffer
        self.counter[:] = 0
        return result
