"""
Created at 23.04.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.product import MomentProduct
import numpy as np


class ParticlesSizeSpectrum(MomentProduct):
    def __init__(self, particles_builder):
        super().__init__(
            particles=particles_builder.particles,
            shape=particles_builder.particles.mesh.grid,
            name='Particles Size Spectrum',
            unit='cm-3',  # TODO!!!
            description='Particles size spectrum',  # TODO
            scale='linear',
            range=[20, 50]
        )
        self.moment_0 = particles_builder.particles.backend.array(1, dtype=int)
        self.moments = particles_builder.particles.backend.array((1, 1), dtype=float)

    def get(self, v_bins):
        vals = np.empty(len(v_bins) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(attr='volume', rank=0, attr_range=(v_bins[i], v_bins[i + 1]))
            vals[i] = self.buffer[0]
        return vals
