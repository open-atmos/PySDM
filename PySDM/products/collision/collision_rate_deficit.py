from PySDM.products.product import Product


class CollisionRateDeficit(Product):

    def __init__(self):
        super().__init__(
            name='collision_rate_deficit',
            description='Collision rate deficit'
        )
        self.collision = None

    def register(self, builder):
        super().register(builder)
        self.collision = self.core.dynamics['Collision']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.collision.collision_rate_deficit)
        self.collision.collision_rate_deficit[:] = 0
        return self.buffer
