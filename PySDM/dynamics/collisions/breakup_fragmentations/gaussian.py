"""
P(x) = exp(-(x-mu)^2 / 2 sigma^2)
"""


class Gaussian:
    def __init__(self, mu, scale):
        self.particulator = None
        self.mu = mu
        self.scale = scale
        self.max_size = None
        self.frag_size = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("radius")
        self.max_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.frag_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, output, u01, is_first_in_pair):
        self.max_size.max(self.particulator.attributes["radius"], is_first_in_pair)
        self.particulator.backend.gauss_fragmentation(
            output, self.mu, self.scale, self.frag_size, self.max_size, u01
        )
