from PySDM.products.product import Product
import time


class _Timer(Product):

    def __init__(self, name, description):
        super().__init__(
            name=name,
            unit='s',
            description=description
        )
        self._time = None
        self.reset()

    def reset(self):
        self._time = self.clock()

    def register(self, builder):
        super().register(builder)
        self.shape = ()

    def get(self) -> float:
        result = -self._time
        self.reset()
        result += self._time
        return result

    def clock(self):
        raise NotImplementedError()


class CPUTime(_Timer):
    def __init__(self):
        super().__init__('cpu_time', 'CPU Time')

    @staticmethod
    def clock():
        return time.process_time()


class WallTime(_Timer):
    def __init__(self):
        super().__init__('wall_time', 'Wall Time')

    @staticmethod
    def clock():
        return time.perf_counter()
