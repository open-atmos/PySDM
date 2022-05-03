"""
ice water content products (mixing ratio and density)
"""
import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class IceWaterContent(MomentProduct):
    def __init__(self, unit="kg/m^3", name=None, __specific=False):
        super().__init__(unit=unit, name=name)
        self.specific = __specific

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="volume", rank=1, filter_range=(-np.inf, 0)
        )
        result = self.buffer.copy()

        self._download_moment_to_buffer(
            attr="volume", rank=0, filter_range=(-np.inf, 0)
        )
        conc = self.buffer

        result[:] *= -self.formulae.constants.rho_i * conc / self.particulator.mesh.dv

        if self.specific:
            self._download_to_buffer(self.particulator.environment["rhod"])
            result[:] /= self.buffer
        return result


class SpecificIceWaterContent(IceWaterContent):
    def __init__(self, unit="kg/kg", name=None, __specific=True):
        super().__init__(unit=unit, name=name)
