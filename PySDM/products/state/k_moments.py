
"""
Created at 10.7.2020 by edejong
"""

from PySDM.products.product import MomentProduct
import numpy as np


class KMoments(MomentProduct):

    def __init__(self):
        super().__init__(
            name='M0-Mk',
            unit='unit (mass or vol)**k',
            description='first k moments of distribution',
        )
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = builder.core.backend.Storage.empty(1, dtype=int)
        self.moments = builder.core.backend.Storage.empty((1, 1), dtype=float)

    def get(self, k):
        vals = np.empty(k)
        for i in range(k):
            if (i == 0):
                self.download_moment_to_buffer(attr='volume', rank=0)
                vals[i] = self.buffer[0]
            else:
                self.download_moment_to_buffer(attr='volume', rank=i)
                vals[i] = self.buffer[0]
                self.download_moment_to_buffer(attr='volume', rank=0)
                vals[i] *= self.buffer[0]
        return vals