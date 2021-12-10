from PySDM.physics import constants as const
from PySDM.products.impl.moment_product import MomentProduct


class MeanRadius(MomentProduct):
    def __init__(self, name=None, unit='m'):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer('volume', rank=1 / 3)
        self.buffer[:] /= const.PI_4_3 ** (1 / 3)
        return self.buffer
