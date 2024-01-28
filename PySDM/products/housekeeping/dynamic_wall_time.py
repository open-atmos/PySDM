"""
wall-time for a given dynamic (fetching a value resets the counter)
"""

from PySDM.products.impl.product import Product


class DynamicWallTime(Product):
    def __init__(self, dynamic, name=None, unit="s"):
        super().__init__(name=name, unit=unit)
        self.value = 0
        self.dynamic = dynamic

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.shape = ()

    def _impl(self, **kwargs):
        tmp = self.value
        self.value = 0
        return tmp

    def notify(self):
        self.value += self.particulator.timers[self.dynamic].time
