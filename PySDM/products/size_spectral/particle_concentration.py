"""
concentration of particles within a grid cell (either per-volume of per-mass-of-dry air,
 optionally restricted to a given size range)
"""
import numpy as np
from PySDM.products.impl.moment_product import MomentProduct


class ParticleConcentration(MomentProduct):
    def __init__(self, radius_range=(0, np.inf), specific=False, name=None, unit='m^-3'):
        self.radius_range = radius_range
        self.specific = specific
        super().__init__(name=name, unit=unit)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            'volume', rank=0,
            filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                          self.formulae.trivia.volume(self.radius_range[1])))
        self.buffer[:] /= self.particulator.mesh.dv
        if self.specific:
            result = self.buffer.copy()  # TODO #217
            self._download_to_buffer(self.particulator.environment['rhod'])
            result[:] /= self.buffer
        else:
            result = self.buffer
        return result


class ParticleSpecificConcentration(ParticleConcentration):
    def __init__(self, radius_range=(0, np.inf), name=None, unit='kg^-1'):
        super().__init__(radius_range=radius_range, specific=True, name=name, unit=unit)
