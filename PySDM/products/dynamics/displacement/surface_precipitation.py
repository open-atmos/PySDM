from PySDM.products.product import Product
from PySDM.physics.constants import rho_w, convert_to, si


class SurfacePrecipitation(Product):

    def __init__(self):
        super().__init__(
            name='surf_precip',
            unit='mm/day',
            description='Surface precipitation'
        )
        self.displacement = None
        self._reset_counters()

    def _reset_counters(self):
        self.accumulated_rainfall = 0.
        self.elapsed_time = 0.

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)
        self.shape = ()
        self.displacement = self.core.dynamics['Displacement']
        self.dv = self.core.mesh.dv
        self.dz = self.core.mesh.dz

    def get(self) -> float:
        if self.elapsed_time == 0.:
            return 0.

        result = rho_w * self.accumulated_rainfall / self.elapsed_time / (self.dv / self.dz)
        self._reset_counters()
        convert_to(result, si.mm / si.day)
        return result

    def notify(self):
        self.accumulated_rainfall += self.displacement.precipitation_in_last_step
        self.elapsed_time += self.displacement.core.dt

