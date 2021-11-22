from PySDM.products.impl.product import Product


class CoalescenceTimestepMin(Product):
    def __init__(self, unit='s', name=None):
        super().__init__(unit=unit, name=name)
        self.coalescence = None
        self.range = None

    def register(self, builder):
        super().register(builder)
        self.coalescence = self.particulator.dynamics['Coalescence']
        self.range = self.coalescence.dt_coal_range

    def _impl(self, **kwargs):
        self._download_to_buffer(self.coalescence.stats_dt_min)
        self.coalescence.stats_dt_min[:] = self.coalescence.particulator.dt
        return self.buffer
