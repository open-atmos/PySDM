from PySDM.products.product import Product


class CoalescenceRate(Product):

    def __init__(self):
        super().__init__(
            name='coalescence_rate',
            description='Coalescence rate'
        )
        self.collision = None

    def register(self, builder):
        super().register(builder)
        self.collision = self.core.dynamics['Collision']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.collision.coalescence_rate)
        self.collision.coalescence_rate[:] = 0
        return self.buffer
