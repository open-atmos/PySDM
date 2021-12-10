from PySDM.products.product import Product


class CollisionRate(Product):

    def __init__(self):
        super().__init__(
            name='collision_rate',
            description='Collision rate'
        )
        self.collision = None

    def register(self, builder):
        super().register(builder)
        self.collision = self.particulator.dynamics['Collision']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.collision.collision_rate)
        self.collision.collision_rate[:] = 0
        return self.buffer
