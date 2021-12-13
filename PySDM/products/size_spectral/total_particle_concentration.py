"""
particle concentration (per volume of air)
"""
from PySDM.products.impl.moment_product import MomentProduct


class TotalParticleConcentration(MomentProduct):
    def __init__(self, name=None, unit='m^-3'):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer('volume', rank=0)
        self.buffer[:] /= self.particulator.mesh.dv
        return self.buffer
