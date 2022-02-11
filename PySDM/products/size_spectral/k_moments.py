
"""
Created at 10.7.2020 by edejong
"""
import numpy as np
from PySDM.products.impl.moment_product import MomentProduct


class KMoments(MomentProduct):

    def __init__(self):
        super().__init__(
            name=None,
            unit='unit (mass or vol)**k',
        )
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = builder.particulator.backend.Storage.empty(1, dtype=int)
        self.moments = builder.particulator.backend.Storage.empty((1, 1), dtype=float)

    def _impl(self, **kwargs):
        k = kwargs['k']
        vals = np.empty(k)
        for i in range(k):
            if i == 0:
                self._download_moment_to_buffer(attr='volume', rank=0)
                vals[i] = self.buffer[0]
            else:
                self._download_moment_to_buffer(attr='volume', rank=i)
                vals[i] = self.buffer[0]
                self._download_moment_to_buffer(attr='volume', rank=0)
                vals[i] *= self.buffer[0]
        return vals
