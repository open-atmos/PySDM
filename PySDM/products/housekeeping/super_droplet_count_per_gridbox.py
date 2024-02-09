"""
super-droplet count per gridbox (dimensionless)
"""

import numba

from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.products.impl.product import Product


class SuperDropletCountPerGridbox(Product):
    def __init__(self, unit="dimensionless", name=None):
        super().__init__(unit=unit, name=name)
        self._jit_impl = None

    def register(self, builder):
        super().register(builder)

        @numba.njit(**{**JIT_FLAGS, "fastmath": builder.formulae.fastmath})
        def jit_impl(cell_start, ravelled_buffer):
            n_cell = cell_start.shape[0] - 1
            for i in numba.prange(n_cell):  # pylint: disable=not-an-iterable
                ravelled_buffer[i] = cell_start[i + 1] - cell_start[i]

        self._jit_impl = jit_impl

    def _impl(self, **kwargs):
        self._jit_impl(
            cell_start=self.particulator.attributes.cell_start.to_ndarray(),
            ravelled_buffer=self.buffer.ravel(),
        )
        return self.buffer
