"""
CPU- and wall-time counters (fetching a value resets the counter)
"""

import time
from abc import abstractmethod

from PySDM.products.impl import Product, register_product


class _Timer(Product):
    def __init__(self, name, unit):
        super().__init__(name=name, unit=unit)
        self._time = -1
        self.reset()

    def reset(self):
        self._time = self.clock()

    def register(self, builder):
        super().register(builder)
        self.shape = ()

    def _impl(self, **kwargs) -> float:
        result = -self._time
        self.reset()
        result += self._time
        return result

    @staticmethod
    @abstractmethod
    def clock():
        raise NotImplementedError()


@register_product()
class CPUTime(_Timer):
    def __init__(self, name="CPU Time", unit="s"):
        super().__init__(unit=unit, name=name)

    @staticmethod
    def clock():
        return time.process_time()


@register_product()
class WallTime(_Timer):
    def __init__(self, name=None, unit="s"):
        super().__init__(unit=unit, name=name)

    @staticmethod
    def clock():
        return time.perf_counter()
