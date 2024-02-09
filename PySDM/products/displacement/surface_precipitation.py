"""
water volume flux derived from sizes of particles crossing bottom domain boundary
"""

from PySDM.products.impl.product import Product


class SurfacePrecipitation(Product):
    def __init__(self, name=None, unit="m/s"):
        super().__init__(unit=unit, name=name)
        self.displacement = None
        self.dv = None
        self.dz = None
        self._reset_counters()

    def _reset_counters(self):
        self.accumulated_rainfall = 0.0
        self.elapsed_time = 0.0

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.shape = ()
        self.displacement = self.particulator.dynamics["Displacement"]
        self.dv = self.particulator.mesh.dv
        self.dz = self.particulator.mesh.dz

    def _impl(self, **kwargs) -> float:
        if self.elapsed_time == 0.0:
            return 0.0

        # TODO #708
        result = (
            self.formulae.constants.rho_w
            * self.accumulated_rainfall
            / self.elapsed_time
            / (self.dv / self.dz)
        )
        self._reset_counters()
        return result

    def notify(self):
        self.accumulated_rainfall += self.displacement.precipitation_in_last_step
        self.elapsed_time += self.displacement.particulator.dt
