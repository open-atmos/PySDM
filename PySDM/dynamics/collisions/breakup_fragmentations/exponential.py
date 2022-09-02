"""
P(x) = exp(-x / lambda); specified in volume units
"""
# TODO #796: introduce common code with Feingold fragmentation, including possible limiter


class ExponFrag:
    def __init__(self, scale, vmin=0.0, nfmax=None):
        self.particulator = None
        self.scale = scale
        self.vmin = vmin
        self.nfmax = nfmax
        self.max_size = None
        self.sum_of_volumes = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.max_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, nf, frag_size, u01, is_first_in_pair):
        self.max_size.max(self.particulator.attributes["volume"], is_first_in_pair)
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.particulator.backend.exp_fragmentation(
            n_fragment=nf,
            scale=self.scale,
            frag_size=frag_size,
            v_max=self.max_size,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
