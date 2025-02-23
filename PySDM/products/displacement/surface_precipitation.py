"""
water (equivalent) volume flux derived from sizes of particles crossing bottom domain boundary
(computed from mass of both liquid and ice water)
"""

from PySDM.products.impl import Product, register_product


@register_product()
class SurfacePrecipitation(Product):
    def __init__(self, name=None, unit="m/s"):
        super().__init__(unit=unit, name=name)
        self.displacement = None
        self.domain_bottom_surface_area = None
        self._reset_counters()

    def _reset_counters(self):
        self.accumulated_rainfall_mass = 0.0
        self.elapsed_time = 0.0

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.shape = ()
        self.displacement = self.particulator.dynamics["Displacement"]
        self.domain_bottom_surface_area = (
            self.particulator.mesh.domain_bottom_surface_area
        )

    def _impl(self, **kwargs) -> float:
        if self.elapsed_time == 0.0:
            return 0.0

        result = (
            self.accumulated_rainfall_mass
            / self.formulae.constants.rho_w
            / self.elapsed_time
            / self.domain_bottom_surface_area
        )
        self._reset_counters()
        return result

    def notify(self):
        self.accumulated_rainfall_mass += (
            self.displacement.precipitation_mass_in_last_step
        )
        self.elapsed_time += self.displacement.particulator.dt
