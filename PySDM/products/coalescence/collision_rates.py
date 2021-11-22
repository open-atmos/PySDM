from PySDM.products.impl.product import Product


class _CollisionRateProduct(Product):
    def __init__(self, name, unit, counter):
        super().__init__(name=name, unit=unit)
        self.timestep_count = 0
        self.counter = counter

    def register(self, builder):
        super().register(builder)
        self.counter = getattr(self.particulator.dynamics['Coalescence'], self.counter)
        self.particulator.observers.append(self)

    def notify(self):
        self.timestep_count += 1

    def _impl(self, **kwargs):
        self._download_to_buffer(self.counter)
        if self.timestep_count != 0:
            self.buffer[:] /= self.timestep_count * self.particulator.dt
        self.counter[:] = 0
        return self.buffer


class CollisionRateDeficitPerGridbox(_CollisionRateProduct):
    def __init__(self, name=None, unit='s^-1'):
        super().__init__(name=name, unit=unit, counter='collision_rate_deficit')


class CollisionRatePerGridbox(_CollisionRateProduct):
    def __init__(self, name=None, unit='s^-1'):
        super().__init__(name=name, unit=unit, counter='collision_rate')
