import numpy as np
from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class ParticlesVolumeSpectrum(SpectrumMomentProduct):
    def __init__(self, radius_bins_edges, name=None, unit='dimensionless'):
        super().__init__(name=name, unit=unit)
        self.radius_bins_edges = radius_bins_edges
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        builder.request_attribute('volume')

        volume_bins_edges = builder.particulator.formulae.trivia.volume(self.radius_bins_edges)
        self.attr_bins_edges = builder.particulator.backend.Storage.from_ndarray(volume_bins_edges)

        super().register(builder)

        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self._recalculate_spectrum_moment(attr='volume', rank=1, filter_attr='volume')

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=1, bin_number=i)
            vals[:, i] = self.buffer.ravel()
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] *= self.buffer.ravel()

        vals *= 1 / np.diff(np.log(self.radius_bins_edges)) / self.particulator.mesh.dv
        return vals
