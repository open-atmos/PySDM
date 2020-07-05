"""
Created at 8.06.2020

@author: Piotr Bartman
@author: Sylwester Arabas
@author: Grzegorz Łazarski
"""

from PySDM.product import MomentProduct
import numpy as np


class ParticlesDrySizeSpectrum(MomentProduct):
    def __init__(self, particles_builder):
        super().__init__(
            particles=particles_builder.particles,
            shape=particles_builder.particles.mesh.grid,
            name='Dry Radius Spectrum',
            unit='cm-3',  # TODO!!!
            description='Spectrum of tha particles\' dry radii',  # TODO
            scale='linear',
            range=[20, 50]
        )
        self.moment_0 = particles_builder.particles.backend.array(1, dtype=int)
        self.moments = particles_builder.particles.backend.array(
            (1, 1), dtype=float)

    def download_moment_to_buffer(self, attr, rank, attr_range=(-np.inf, np.inf)):
        self.particles.state.moments(self.moment_0, self.moments, {
                                     attr: (rank,)}, attr_name=attr, attr_range=attr_range)
        if rank == 0:  # TODO
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(
                self.particles.backend.read_row(self.moments, 0))

    def get(self, v_bins):
        vals = np.empty(len(v_bins) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(
                attr='dry volume', rank=0, attr_range=(v_bins[i], v_bins[i + 1]))
            vals[i] = self.buffer[0]
        return vals
