"""
P(m; x, y) = nu^2 * (x+y) exp(-m * nu)
nu = 1/m* where m* is a scaling factor for fragment size dist.
"""


class Feingold1988Frag:
    def __init__(self, scale, fragtol=1e-3, vmin=0.0, nfmax=None):
        self.particulator = None
        self.scale = scale
        self.fragtol = fragtol
        self.vmin = vmin
        self.nfmax = nfmax
        self.max_size = None
        self.frag_size = None
        self.sum_of_volumes = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
        self.max_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.frag_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, output, u01, is_first_in_pair):
        self.max_size.max(self.particulator.attributes["volume"], is_first_in_pair)
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.particulator.backend.feingold1988_fragmentation(
            output,
            self.scale,
            self.frag_size,
            self.max_size,
            self.sum_of_volumes,
            u01,
            self.fragtol,
            self.vmin,
            self.nfmax,
        )
