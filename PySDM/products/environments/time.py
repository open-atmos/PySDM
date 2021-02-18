from PySDM.products.product import Product


class Time(Product):

    def __init__(self):
        super().__init__(
            name='t',
            unit='s',
            description='Time'
        )
        self.t = 0

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)

    def get(self):
        return self.t

    def notify(self):
        self.t += self.core.dt
