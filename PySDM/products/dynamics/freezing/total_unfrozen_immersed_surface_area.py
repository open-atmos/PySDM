from ...product import MomentProduct
import numpy as np
from PySDM.physics.constants import convert_to, si


class TotalUnfrozenImmersedSurfaceArea(MomentProduct):
    def __init__(self):
        super().__init__(
            name='A_tot',
            description='total unfrozen immersed surface area',
            unit='m2'
        )

    def get(self):
        params = {
            'attr': 'immersed surface area',
            'filter_attr': 'volume',
            'filter_range': (0, np.inf)
        }
        self.download_moment_to_buffer(**params, rank=1)
        result = np.copy(self.buffer)
        self.download_moment_to_buffer(**params, rank=0)
        result[:] *= self.buffer
        return result
