"""
Created at 23.04.2020
"""

from PySDM.product import MomentProduct
import numpy as np


class ParticlesSizeSpectrum(MomentProduct):
    def __init__(self, particles_builder, v_bins):
        self.v_bins = v_bins
        super().__init__(
            particles=particles_builder.particles,
            shape=(*particles_builder.particles.mesh.grid, len(self.v_bins) - 1),
            name='Particles Size Spectrum',
            unit='cm-3',  # TODO!!!
            description='Particles size spectrum',  # TODO
            scale='linear',
            range=[20, 50]
        )

    def get(self):
        vals = np.empty([self.particles.mesh.n_cell, len(self.v_bins) - 1])
        for i in range(len(self.v_bins) - 1):
            self.download_moment_to_buffer(attr='volume', rank=0, attr_range=(self.v_bins[i], self.v_bins[i + 1]))
            vals[:, i] = self.buffer.ravel()
        return np.squeeze(vals.reshape(self.shape))
