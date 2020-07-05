"""
Created at 23.04.2020
"""

from PySDM.product import MomentProduct
import numpy as np
from PySDM.physics import constants as const
from PySDM.physics import formulae as phys


class ParticlesSizeSpectrum(MomentProduct):

    def __init__(self, particles_builder, v_bins, normalise_by_dv=False):
        self.v_bins = v_bins
        self.normalise_by_dv = normalise_by_dv
        super().__init__(
            core=particles_builder.core,
            shape=(*particles_builder.core.mesh.grid, len(self.v_bins) - 1),
            name='Particles Size Spectrum',
            unit=f"mg-1 μm-1{'' if normalise_by_dv else ' m^3'}",
            description='Specific concentration density',
            scale='linear',
            range=[20, 50]
        )

    def get(self):
        vals = np.empty([self.particles.mesh.n_cell, len(self.v_bins) - 1])
        for i in range(len(self.v_bins) - 1):
            self.download_moment_to_buffer(
                attr='volume', rank=0, attr_range=(self.v_bins[i], self.v_bins[i + 1])
            )
            vals[:, i] = self.buffer.ravel()

        if self.normalise_by_dv:
            vals[:] /= self.particles.mesh.dv

        self.download_to_buffer(self.particles.environment['rhod'])
        rhod = self.buffer.ravel()
        for i in range(len(self.v_bins) - 1):
            dr = phys.radius(volume=self.v_bins[i + 1]) - phys.radius(volume=self.v_bins[i])
            vals[:, i] /= rhod * dr

        const.convert_to(vals, const.si.micrometre**-1 * const.si.milligram**-1)

        return np.squeeze(vals.reshape(self.shape))
