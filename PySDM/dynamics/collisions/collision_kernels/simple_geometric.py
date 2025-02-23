"""
basic geometric kernel (not taking fall velocity into account)
"""


class SimpleGeometric:
    def __init__(self, C):
        self.particulator = None
        self.pair_tmp = None
        self.C = C

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("radius")
        builder.request_attribute("area")
        self.pair_tmp = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, output, is_first_in_pair):
        output[:] = self.C
        self.pair_tmp.sum(self.particulator.attributes["radius"], is_first_in_pair)
        self.pair_tmp **= 2
        output *= self.pair_tmp
        self.pair_tmp.distance(self.particulator.attributes["area"], is_first_in_pair)
        output *= self.pair_tmp
