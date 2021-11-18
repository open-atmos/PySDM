"""
P(x) = exp(-(x-mu)^2 / 2 sigma^2)
"""

import numpy as np

class Gaussian:

    def __init__(self, mu, scale):
        self.core = None
        self.mu = mu
        self.scale = scale
        
    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('radius')
        self.max_size = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.frag_size = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)

    def __call__(self, output, u01, is_first_in_pair):
        self.max_size.max(self.core.particles['radius'],is_first_in_pair)
        self.core.backend.gauss_fragmentation(output, self.mu, self.scale, self.frag_size,
            self.max_size, u01)
        