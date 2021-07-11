import numpy as np
from PySDM.products.product import SpectrumMomentProduct


class ParticlesVolumeSpectrum(SpectrumMomentProduct):

    def __init__(self, radius_bins_edges):
        super().__init__(
            name='dv/dlnr',
            unit='1/(unit dr/r)',
            description='Particles volume distribution'
        )
        self.radius_bins_edges = radius_bins_edges
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        builder.request_attribute('volume')

        volume_bins_edges = builder.core.formulae.trivia.volume(self.radius_bins_edges)
        self.attr_bins_edges = builder.core.bck.Storage.from_ndarray(volume_bins_edges)

        super().register(builder)

        self.shape = (*builder.core.mesh.grid, len(self.attr_bins_edges) - 1)

    def get(self):
        vals = np.empty([self.core.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self.recalculate_spectrum_moment(attr=f'volume', rank=1, filter_attr='volume')

        for i in range(vals.shape[1]):
            self.download_spectrum_moment_to_buffer(rank=1, bin_number=i)
            vals[:, i] = self.buffer.ravel()
            self.download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] *= self.buffer.ravel()

        vals *= 1 / np.diff(np.log(self.radius_bins_edges)) / self.core.mesh.dv
        return vals
