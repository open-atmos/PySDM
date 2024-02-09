"""
Minimal collision timestep used when adaptive timestepping is enabled in the
 `PySDM.dynamics.collisions.collision.Collision` dynamic (fetching a value resets the counter)
"""

from PySDM.products.impl.product import Product


class CollisionTimestepMin(Product):
    def __init__(self, unit="s", name=None):
        super().__init__(unit=unit, name=name)
        self.collision = None
        self.range = None

    def register(self, builder):
        super().register(builder)
        self.collision = self.particulator.dynamics["Collision"]
        self.range = self.collision.dt_coal_range

    def _impl(self, **kwargs):
        self._download_to_buffer(self.collision.stats_dt_min)
        self.collision.stats_dt_min[:] = self.collision.particulator.dt
        return self.buffer
