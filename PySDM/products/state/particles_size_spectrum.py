import numpy as np

from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class ParticlesSizeSpectrum(MomentProduct):

    def __init__(self, v_bins, name, dry=False, normalise_by_dv=False):
        self.volume_attr = 'dry volume' if dry else 'volume'
        self.v_bins = v_bins
        self.normalise_by_dv = normalise_by_dv
        super().__init__(
            name=name,
            unit=f"mg-1 um-1{'' if normalise_by_dv else ' m^3'}",
            description='Specific concentration density'
        )

    def register(self, builder):
        builder.request_attribute(self.volume_attr)
        super().register(builder)
        self.shape = (*builder.core.mesh.grid, len(self.v_bins) - 1)

    def get(self):
        vals = np.empty([self.core.mesh.n_cell, len(self.v_bins) - 1])
        for i in range(len(self.v_bins) - 1):
            self.download_moment_to_buffer(
                attr=self.volume_attr, rank=0, filter_attr=self.volume_attr, filter_range=(self.v_bins[i], self.v_bins[i + 1])
            )
            vals[:, i] = self.buffer.ravel()

        if self.normalise_by_dv:
            vals[:] /= self.core.mesh.dv

        self.download_to_buffer(self.core.environment['rhod'])
        rhod = self.buffer.ravel()
        for i in range(len(self.v_bins) - 1):
            dr = self.formulae.trivia.radius(volume=self.v_bins[i + 1]) - \
                 self.formulae.trivia.radius(volume=self.v_bins[i])
            vals[:, i] /= rhod * dr

        const.convert_to(vals, const.si.micrometre**-1 * const.si.milligram**-1)

        return np.squeeze(vals.reshape(self.shape))


class ParticlesWetSizeSpectrum(ParticlesSizeSpectrum):
    def __init__(self, v_bins, normalise_by_dv=False):
        super().__init__(v_bins, dry=False, normalise_by_dv=normalise_by_dv, name='Particles Wet Size Spectrum')


class ParticlesDrySizeSpectrum(ParticlesSizeSpectrum):
    def __init__(self, v_bins, normalise_by_dv=False):
        super().__init__(v_bins, dry=True, normalise_by_dv=normalise_by_dv, name='Particles Dry Size Spectrum')