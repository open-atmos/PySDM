"""
physical time (in dt increments)
"""

from PySDM.products.impl.product import Product


class Time(Product):
    def __init__(self, name=None, unit="s"):
        super().__init__(name=name, unit=unit)
        self.t = 0

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)

    def _impl(self, **kwargs):
        return self.t

    def notify(self):
        self.t += self.particulator.dt
