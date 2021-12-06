from PySDM.products.product import Product


class BreakupRate(Product):

    def __init__(self):
        super().__init__(
            name='breakup_rate',
            description='Breakup rate'
        )
        self.collision = None

    def register(self, builder):
        super().register(builder)
        self.collision = self.core.dynamics['Collision']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.collision.breakup_rate)
        self.collision.breakup_rate[:] = 0
        return self.buffer
