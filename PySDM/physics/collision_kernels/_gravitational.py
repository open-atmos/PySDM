class Gravitational:

    def __init__(self):
        self.core = None
        self.pair_tmp = None

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('radius')
        builder.request_attribute('terminal velocity')
        self.pair_tmp = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
