class Gravitational:

    def __init__(self):
        self.particulator = None
        self.pair_tmp = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute('radius')
        builder.request_attribute('terminal velocity')
        self.pair_tmp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
