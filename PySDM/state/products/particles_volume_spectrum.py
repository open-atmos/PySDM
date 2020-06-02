"""
Created at 28.04.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import MomentProduct
from PySDM.physics import formulae as phys
import numpy as np


class ParticlesVolumeSpectrum(MomentProduct):
    def __init__(self, particles_builder):
        super().__init__(
            particles=particles_builder.particles,
            shape=particles_builder.particles.mesh.grid,
            name='dv/dlnr',
            unit='1/(unit dr/r)',
            description='Particles volume distribution',
            scale='linear',
            range=[20, 50]
        )
        self.moment_0 = particles_builder.particles.backend.storage.empty(1, dtype=int)
        self.moments = particles_builder.particles.backend.storage.empty((1, 1), dtype=float)

    def get(self, radius_bins_edges):
        volume_bins_edges = phys.volume(radius_bins_edges)
        vals = np.empty(len(volume_bins_edges) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(attr='volume', rank=1,
                                           attr_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] = self.buffer[0]
            self.download_moment_to_buffer(attr='volume', rank=0,
                                           attr_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] *= self.buffer[0]
        vals *= 1 / np.diff(np.log(radius_bins_edges)) / self.particles.mesh.dv
        return vals
