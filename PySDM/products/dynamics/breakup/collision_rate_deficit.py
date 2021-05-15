"""
Created at 14.05.2021 by edejong
"""

from PySDM.products.product import Product


class BreakupRateDeficit(Product):

    def __init__(self):
        super().__init__(
            name='breakup_rate_deficit',
            description='Breakup rate deficit'
        )
        self.breakup = None

    def register(self, builder):
        super().register(builder)
        self.breakup = self.core.dynamics['Breakup']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.breakup.collision_rate_deficit)
        self.breakup.collision_rate_deficit[:] = 0
        return self.buffer
