"""
Based on 10.5194/gmd-9-1455-2016 for ocean particles
"""


class SLAMS:
    def __init__(self, vmin=0.0, nfmax=None):
        self.particulator = None
        self.p_vec = None
        self.max_size = None
        self.sum_of_volumes = None
        self.vmin = vmin
        self.nfmax = nfmax

    def __call__(self, nf, frag_size, u01, is_first_in_pair):
        self.max_size.max(self.particulator.attributes["volume"], is_first_in_pair)
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.particulator.backend.slams_fragmentation(
            n_fragment=nf,
            frag_size=frag_size,
            v_max=self.max_size,
            x_plus_y=self.sum_of_volumes,
            probs=self.p_vec,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )

    def register(self, builder):
        self.particulator = builder.particulator
        self.p_vec = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.max_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
