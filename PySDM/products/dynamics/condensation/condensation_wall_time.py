from ...product import Product


class CondensationWallTime(Product):
    def __init__(self):
        super().__init__(
            name='cond_wall_time',
            unit='s',
            description='Condensation wall time',
        )
        self.value = 0

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)
        self.shape = ()

    def get(self):
        tmp = self.value
        self.value = 0
        return tmp

    def notify(self):
        self.value += self.core.timers['Condensation'].time
