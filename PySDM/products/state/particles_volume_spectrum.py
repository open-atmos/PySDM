import numpy as np
from PySDM.products.product import MomentProduct


class ParticlesVolumeSpectrum(MomentProduct):

    def __init__(self):
        super().__init__(
            name='dv/dlnr',
            unit='1/(unit dr/r)',
            description='Particles volume distribution'
        )
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = builder.core.backend.Storage.empty(1, dtype=int)
        self.moments = builder.core.backend.Storage.empty((1, 1), dtype=float)

    def get(self, radius_bins_edges):
        volume_bins_edges = self.formulae.trivia.volume(radius_bins_edges)
        vals = np.empty(len(volume_bins_edges) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(attr='volume', rank=1,
                                           filter_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] = self.buffer[0]
            self.download_moment_to_buffer(attr='volume', rank=0,
                                           filter_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] *= self.buffer[0]
        vals *= 1 / np.diff(np.log(radius_bins_edges)) / self.core.mesh.dv
        return vals
