from PySDM.products.product import Product


class DynamicWallTime(Product):
    def __init__(self, dynamic):
        super().__init__(
            name=f'{dynamic}_wall_time',
            unit='s',
            description=f'{dynamic} wall time',
        )
        self.value = 0
        self.dynamic = dynamic

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.shape = ()

    def get(self):
        tmp = self.value
        self.value = 0
        return tmp

    def notify(self):
        self.value += self.particulator.timers[self.dynamic].time
