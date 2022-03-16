"""
Based on 10.5194/gmd-9-1455-2016 for ocean particles
"""


class SLAMS:
    def __init__(self):
        self.particulator = None
        self.p_vec = None

    def __call__(self, output, u01, is_first_in_pair):
        self.particulator.backend.slams_fragmentation(output, self.p_vec, u01)

    def register(self, builder):
        self.particulator = builder.particulator
        self.p_vec = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
