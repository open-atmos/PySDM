"""
Created at 23.04.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import MomentProduct
import numpy as np


class ParticlesSizeSpectrum(MomentProduct):
    def __init__(self, particles_builder, v_bins):
        self.shape = particles_builder.particles.mesh.grid
        self.shape.append(len(self.v_bins))
        super().__init__(
            particles=particles_builder.particles,
            shape=self.shape,
            name='Particles Size Spectrum',
            unit='cm-3',  # TODO!!!
            description='Particles size spectrum',  # TODO
            scale='linear',
            range=[20, 50]
        )
        self.moment_0 = particles_builder.particles.backend.array(self.shape, dtype=int)
        self.moments = particles_builder.particles.backend.array(self.shape, dtype=float)
        self.v_bins = v_bins

    def get(self):
        vals = np.empty(len(self.v_bins) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(attr='volume', rank=0, attr_range=(self.v_bins[i], self.v_bins[i + 1]))
            vals[i] = self.buffer[0]
        return vals
