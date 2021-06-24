import numpy as np

from PySDM.physics import constants as const


from PySDM.products.product import SpectrumMomentProduct


class ParticlesSizeSpectrum(SpectrumMomentProduct):

    def __init__(self, radius_bins_edges, name, dry=False, normalise_by_dv=False):
        self.volume_attr = 'dry volume' if dry else 'volume'
        self.radius_bins_edges = radius_bins_edges
        self.normalise_by_dv = normalise_by_dv
        super().__init__(
            name=name,
            unit=f"mg-1 um-1{'' if normalise_by_dv else ' m^3'}",
            description='Specific concentration density'
        )

    def register(self, builder):
        builder.request_attribute(self.volume_attr)

        volume_bins_edges = builder.core.formulae.trivia.volume(self.radius_bins_edges)
        self.attr_bins_edges = builder.core.bck.Storage.from_ndarray(volume_bins_edges)

        super().register(builder)

        self.shape = (*builder.core.mesh.grid, len(self.attr_bins_edges) - 1)

    def get(self):
        vals = np.empty([self.core.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self.recalculate_spectrum_moment(attr=self.volume_attr, rank=1, filter_attr=self.volume_attr)

        for i in range(vals.shape[1]):
            self.download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        if self.normalise_by_dv:
            vals[:] /= self.core.mesh.dv

        self.download_to_buffer(self.core.environment['rhod'])
        rhod = self.buffer.ravel()
        for i in range(len(self.attr_bins_edges) - 1):
            dr = self.formulae.trivia.radius(volume=self.attr_bins_edges[i + 1]) - \
                 self.formulae.trivia.radius(volume=self.attr_bins_edges[i])
            vals[:, i] /= rhod * dr

        const.convert_to(vals, const.si.micrometre**-1 * const.si.milligram**-1)

        return np.squeeze(vals.reshape(self.shape))


class ParticlesWetSizeSpectrum(ParticlesSizeSpectrum):
    def __init__(self, radius_bins_edges, normalise_by_dv=False):
        super().__init__(radius_bins_edges, dry=False, normalise_by_dv=normalise_by_dv, name='Particles Wet Size Spectrum')


class ParticlesDrySizeSpectrum(ParticlesSizeSpectrum):
    def __init__(self, radius_bins_edges, normalise_by_dv=False):
        super().__init__(radius_bins_edges, dry=True, normalise_by_dv=normalise_by_dv, name='Particles Dry Size Spectrum')