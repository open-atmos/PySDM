"""
TODO #744
"""


class Gravitational:  # pylint: disable=too-few-public-methods
    def __init__(self, relax_velocity=False):
        self.particulator = None
        self.pair_tmp = None
        self.relax_velocity = relax_velocity
        self.vel_attr = "fall velocity" if relax_velocity else "terminal velocity"

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("radius")
        builder.request_attribute(self.vel_attr)
        self.pair_tmp = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
