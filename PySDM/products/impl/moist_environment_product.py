from PySDM.environments._moist import _Moist
from PySDM.products.impl.product import Product


class MoistEnvironmentProduct(Product):
    def __init__(self, *, name, unit, var=None):
        super().__init__(name=name, unit=unit)
        self.environment = None
        self.source = None
        self.var = var or name

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.environment = builder.particulator.environment
        self.source = self.environment[self.var]

    def notify(self):
        if isinstance(self.environment, _Moist):
            self.source = self.environment.get_predicted(self.var)

    def _impl(self, **kwargs):
        self._download_to_buffer(self.source)
        return self.buffer
