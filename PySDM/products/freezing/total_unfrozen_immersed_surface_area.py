"""
mean immersed surface area per particle for unfrozen particles
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class TotalUnfrozenImmersedSurfaceArea(MomentProduct):
    def __init__(self, unit="m^2", name=None):
        super().__init__(unit=unit, name=name)

    def _impl(self, **kwargs):
        params = {
            "attr": "immersed surface area",
            "filter_attr": "volume",
            "filter_range": (0, np.inf),
        }
        self._download_moment_to_buffer(**params, rank=1)
        result = np.copy(self.buffer)
        self._download_moment_to_buffer(**params, rank=0)
        result[:] *= self.buffer
        # TODO #599 per volume / per gridbox ?
        return result
