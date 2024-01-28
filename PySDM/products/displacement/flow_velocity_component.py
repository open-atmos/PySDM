"""
reports on the maximum Courant field value for each cell (maximum of
absolute values of Courant number on all edges of a cell)
"""

import numpy as np

from PySDM.products.impl.product import Product


class FlowVelocityComponent(Product):
    def __init__(self, component: int, name=None, unit="m/s"):
        super().__init__(unit=unit, name=name)
        assert component in (0, 1)
        self.component = component
        self.displacement = None
        self.grid_step = np.nan
        self.time_step = np.nan

    def register(self, builder):
        super().register(builder)
        self.displacement = self.particulator.dynamics["Displacement"]
        self.time_step = self.particulator.dt
        mesh = self.particulator.mesh
        self.grid_step = mesh.size[self.component] / mesh.grid[self.component]

    def _impl(self, **kwargs):
        courant_component = self.displacement.courant[self.component].to_ndarray()
        if self.component == 0:
            self.buffer[:] = 0.5 * (
                courant_component[:-1, :] + courant_component[1:, :]
            )
        elif self.component == 1:
            self.buffer[:] = 0.5 * (
                courant_component[:, :-1] + courant_component[:, 1:]
            )
        else:
            raise NotImplementedError()
        self.buffer[:] *= self.grid_step / self.time_step
        return self.buffer
