"""
super-droplet count per gridbox (dimensionless)
"""
from PySDM.products.impl.product import Product


class SuperDropletCountPerGridbox(Product):
    def __init__(self, unit='dimensionless', name=None):
        super().__init__(unit=unit, name=name)

    def _impl(self, **kwargs):
        cell_start = self.particulator.attributes.cell_start
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            self.buffer.ravel()[i] = cell_start[i + 1] - cell_start[i]
        return self.buffer
