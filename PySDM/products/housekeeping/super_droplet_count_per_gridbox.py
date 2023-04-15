"""
super-droplet count per gridbox (dimensionless)
"""
import numba

from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.products.impl.product import Product


class SuperDropletCountPerGridbox(Product):
    def __init__(self, unit="dimensionless", name=None):
        super().__init__(unit=unit, name=name)
        self.impl = None

    def register(self, builder):
        super().register(builder)

        @numba.njit(**{**JIT_FLAGS, "fastmath": builder.formulae.fastmath})
        def impl(cell_start, buffer_ravel):
            n_cell = cell_start.shape[0] - 1
            for i in numba.prange(n_cell):  # pylint: disable=not-an-iterable
                buffer_ravel[i] = cell_start[i + 1] - cell_start[i]

        self.impl = impl

    def _impl(self, **kwargs):
        self.impl(
            cell_start=self.particulator.attributes.cell_start.to_ndarray(),
            buffer_ravel=self.buffer.ravel(),
        )
        return self.buffer
