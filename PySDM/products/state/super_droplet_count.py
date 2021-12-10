from PySDM.products.product import Product


class SuperDropletCount(Product):

    def __init__(self):
        super().__init__(
            name='n_sd',
            unit='#/gridbox',
            description='Super droplet count'
        )

    def get(self):
        cell_start = self.particulator.attributes.cell_start
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            self.buffer.ravel()[i] = cell_start[i + 1] - cell_start[i]
        return self.buffer
