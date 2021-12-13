"""
particle specific concentration (per mass of dry air)
"""
from PySDM.products.impl.moment_product import MomentProduct


class TotalParticleSpecificConcentration(MomentProduct):
    def __init__(self, name=None, unit='kg^-1'):
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer('volume', rank=0)
        result = self.buffer.copy()  # TODO #217
        self._download_to_buffer(self.particulator.environment['rhod'])
        result[:] /= self.particulator.mesh.dv
        result[:] /= self.buffer
        return result
