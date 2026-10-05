"""common parent class for gravitational collision kernels"""


class Gravitational:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.particulator = None
        self.pair_tmp = None

    def register(self, particulator):
        self.particulator = particulator
        particulator.request_attribute("radius")
        particulator.request_attribute("relative fall velocity")
        self.pair_tmp = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
