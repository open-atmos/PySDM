"""
reports on the maximum Courant field value for each cell (maximum of
absolute values of Courant number on all edges of a cell)
"""

import numpy as np

from PySDM.products.impl.product import Product


class MaxCourantNumber(Product):
    def __init__(self, name=None, unit="dimensionless"):
        super().__init__(unit=unit, name=name)
        self.displacement = None

    def register(self, builder):
        super().register(builder)
        self.displacement = self.particulator.dynamics["Displacement"]

    def _impl(self, **kwargs):
        self.buffer[:] = 0

        field = tuple(
            abs(component.to_ndarray()) for component in self.displacement.courant
        )
        self.buffer[:] = np.maximum(
            self.buffer, np.maximum(field[0][:-1, :], field[0][1:, :])
        )
        self.buffer[:] = np.maximum(
            self.buffer, np.maximum(field[1][:, :-1], field[1][:, 1:])
        )

        return self.buffer
