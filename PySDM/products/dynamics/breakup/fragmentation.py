"""
Created at 14.05.2021 by edejong
"""

from PySDM.products.product import Product


class BreakupFragments(Product):

    def __init__(self):
        super().__init__(
            name='breakup_n_fragments',
            description='Breakup number of fragments'
        )
        self.breakup = None

    def register(self, builder):
        super().register(builder)
        self.breakup = self.core.dynamics['Breakup']

    def get(self):  # TODO #345 take into account NUMBER of substeps (?)
        self.download_to_buffer(self.breakup.n_fragment)
        self.breakup.n_fragment[:] = 0
        return self.buffer
